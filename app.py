"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Run:  pip install -r requirements.txt
      streamlit run app.py
Place European_Bank_csv_days.csv in the same folder (or upload it in the sidebar).
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Retention Strategy Analytics", layout="wide")

DEFAULT_CSV = "European_Bank_csv_days.csv"
REQUIRED = ["CustomerId", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]


# ---------------------------------------------------------------- data
@st.cache_data
def load(file) -> pd.DataFrame:
    return pd.read_csv(file)


def validate(df: pd.DataFrame) -> list[str]:
    issues = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    if df[REQUIRED].isna().any().any():
        issues.append("Missing values found.")
    if df.CustomerId.duplicated().any():
        issues.append("Duplicate CustomerId values found.")
    for c in ["HasCrCard", "IsActiveMember", "Exited"]:
        if not set(df[c].unique()) <= {0, 1}:
            issues.append(f"{c} is not strictly binary.")
    return issues


def add_features(df: pd.DataFrame, hb_quantile: float) -> tuple[pd.DataFrame, float]:
    df = df.copy()
    hb_cut = df.Balance.quantile(hb_quantile)

    def profile(r):
        if r.IsActiveMember == 0 and r.Balance >= hb_cut:
            return "Inactive High-Balance"
        if r.IsActiveMember == 0:
            return "Inactive Disengaged"
        if r.NumOfProducts == 1:
            return "Active Low-Product"
        return "Active Engaged"

    df["Profile"] = df.apply(profile, axis=1)
    df["Status"] = df.IsActiveMember.map({1: "Active", 0: "Inactive"})
    return df, hb_cut


def rate(s):  # churn % helper
    return s.mean() * 100 if len(s) else float("nan")


# ---------------------------------------------------------------- load
st.title("Customer Engagement & Product Utilization Analytics")
st.caption("Retention strategy dashboard: engagement, product depth and relationship strength")

up = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv")
try:
    raw = load(up if up else DEFAULT_CSV)
except FileNotFoundError:
    st.error(f"'{DEFAULT_CSV}' not found. Upload the CSV in the sidebar.")
    st.stop()

issues = validate(raw)
if issues:
    st.error(" ".join(issues))
    if any("Missing columns" in i for i in issues):
        st.stop()

# ---------------------------------------------------------------- filters
st.sidebar.header("Filters")
hb_q = st.sidebar.slider("High-balance cut-off (percentile)", 0.50, 0.95, 0.75, 0.05)
data, hb_cut = add_features(raw, hb_q)
st.sidebar.caption(f"High balance = {hb_cut:,.0f} or more")

eng = st.sidebar.multiselect("Engagement status", ["Active", "Inactive"], ["Active", "Inactive"])
prof_opts = sorted(data.Profile.unique())
profs = st.sidebar.multiselect("Engagement profile", prof_opts, prof_opts)
p_lo, p_hi = st.sidebar.slider("Number of products", 1, 4, (1, 4))
b_lo, b_hi = st.sidebar.slider("Balance range", 0, int(data.Balance.max()),
                               (0, int(data.Balance.max())), step=5000)
s_lo, s_hi = st.sidebar.slider("Estimated salary range", 0, int(data.EstimatedSalary.max()),
                               (0, int(data.EstimatedSalary.max())), step=5000)
geo = st.sidebar.multiselect("Geography", sorted(data.Geography.unique()), sorted(data.Geography.unique()))
card = st.sidebar.selectbox("Credit card", ["All", "Has card", "No card"])

f = data[
    data.Status.isin(eng) & data.Profile.isin(profs)
    & data.NumOfProducts.between(p_lo, p_hi)
    & data.Balance.between(b_lo, b_hi)
    & data.EstimatedSalary.between(s_lo, s_hi)
    & data.Geography.isin(geo)
]
if card == "Has card":
    f = f[f.HasCrCard == 1]
elif card == "No card":
    f = f[f.HasCrCard == 0]

if f.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ---------------------------------------------------------------- KPIs (computed on filtered data)
act, inact = f[f.IsActiveMember == 1], f[f.IsActiveMember == 0]
err = rate(inact.Exited) / rate(act.Exited) if len(act) and len(inact) and rate(act.Exited) > 0 else float("nan")
hb_inact = f[(f.IsActiveMember == 0) & (f.Balance >= hb_cut)]
hbdr = rate(hb_inact.Exited)
card_ret = (1 - f[f.HasCrCard == 1].Exited.mean()) / (1 - f[f.HasCrCard == 0].Exited.mean()) \
    if f.HasCrCard.nunique() == 2 else float("nan")

# Product Depth Index and RSI use product-tier retention from the FULL dataset
tier_ret = 1 - data.groupby("NumOfProducts").Exited.mean()
single_ret = tier_ret.get(1, float("nan"))
pdi = (tier_ret / single_ret).round(2)
f = f.assign(RSI=100 * f.NumOfProducts.map(tier_ret) * (0.5 + 0.5 * f.IsActiveMember))

k = st.columns(5)
k[0].metric("Customers", f"{len(f):,}")
k[1].metric("Churn rate", f"{rate(f.Exited):.2f}%")
k[2].metric("Engagement Retention Ratio", f"{err:.2f}x" if err == err else "n/a",
            help="Inactive churn / active churn")
k[3].metric("High-Balance Disengagement Rate", f"{hbdr:.2f}%" if hbdr == hbdr else "n/a",
            help="Churn of inactive customers with balance above the cut-off")
k[4].metric("Card Stickiness Score", f"{card_ret:.2f}" if card_ret == card_ret else "n/a",
            help="Retention (card holders) / retention (non-holders). ~1 = no effect")

tabs = st.tabs(["Engagement vs churn", "Product utilization", "High-value disengaged",
                "Retention strength", "Data & validation"])

# ---------------------------------------------------------------- tab 1
with tabs[0]:
    c1, c2 = st.columns(2)
    by_status = f.groupby("Status").Exited.agg(["size", rate]).reset_index()
    by_status.columns = ["Status", "Customers", "Churn %"]
    c1.plotly_chart(px.bar(by_status, x="Status", y="Churn %", text="Churn %", color="Status",
                           title="Churn rate: active vs inactive").update_traces(texttemplate="%{text:.1f}%"),
                    use_container_width=True)
    by_prof = f.groupby("Profile").Exited.agg(["size", "sum", rate]).reset_index()
    by_prof.columns = ["Profile", "Customers", "Churned", "Churn %"]
    c2.plotly_chart(px.bar(by_prof, x="Profile", y="Churn %", text="Churn %", color="Profile",
                           title="Churn rate by engagement profile").update_traces(texttemplate="%{text:.1f}%"),
                    use_container_width=True)
    by_prof["Share of churn %"] = by_prof.Churned / by_prof.Churned.sum() * 100
    st.dataframe(by_prof.round(2), hide_index=True, use_container_width=True)

    g = f.groupby(["Geography", "Status"]).Exited.apply(rate).reset_index(name="Churn %")
    st.plotly_chart(px.bar(g, x="Geography", y="Churn %", color="Status", barmode="group",
                           title="Churn by geography and engagement status"), use_container_width=True)

# ---------------------------------------------------------------- tab 2
with tabs[1]:
    c1, c2 = st.columns(2)
    bp = f.groupby("NumOfProducts").Exited.agg(["size", rate]).reset_index()
    bp.columns = ["Products", "Customers", "Churn %"]
    c1.plotly_chart(px.bar(bp, x="Products", y="Churn %", text="Churn %",
                           title="Churn rate by number of products").update_traces(texttemplate="%{text:.1f}%"),
                    use_container_width=True)
    bpa = f.groupby(["NumOfProducts", "Status"]).Exited.apply(rate).reset_index(name="Churn %")
    c2.plotly_chart(px.bar(bpa, x="NumOfProducts", y="Churn %", color="Status", barmode="group",
                           title="Products x engagement"), use_container_width=True)
    f["Product group"] = f.NumOfProducts.map(lambda n: "Single product" if n == 1 else "Multi-product (2-4)")
    sm = f.groupby("Product group").Exited.agg(["size", rate]).reset_index()
    sm.columns = ["Group", "Customers", "Churn %"]
    st.subheader("Single vs multi-product")
    st.dataframe(sm.round(2), hide_index=True, use_container_width=True)
    st.subheader("Product Depth Index")
    st.caption("Retention of each product tier relative to single-product retention (full dataset).")
    st.dataframe(pdi.rename("Product Depth Index").reset_index().rename(columns={"NumOfProducts": "Products"}),
                 hide_index=True, use_container_width=True)
    st.info("Retention improves from 1 to 2 products, then collapses at 3-4. "
            "Do not read multi-product as uniformly protective.")

# ---------------------------------------------------------------- tab 3
with tabs[2]:
    st.subheader("High-value disengaged customer detector")
    c1, c2, c3 = st.columns(3)
    min_bal = c1.number_input("Minimum balance", 0, int(data.Balance.max()), int(hb_cut), step=5000)
    max_sal = c2.number_input("Maximum salary (0 = no limit)", 0, int(data.EstimatedSalary.max()), 0, step=5000)
    only_retained = c3.checkbox("Only customers who have not yet churned", value=True)

    det = data[(data.IsActiveMember == 0) & (data.Balance >= min_bal)]
    if max_sal:
        det = det[det.EstimatedSalary <= max_sal]
    if only_retained:
        det = det[det.Exited == 0]
    det = det.assign(**{"Balance/Salary": (det.Balance / det.EstimatedSalary).round(2)})

    m = st.columns(3)
    m[0].metric("Customers flagged", f"{len(det):,}")
    m[1].metric("Balance at risk", f"{det.Balance.sum():,.0f}")
    pool = data[(data.IsActiveMember == 0) & (data.Balance >= min_bal)]
    m[2].metric("Historical churn of this group", f"{rate(pool.Exited):.2f}%")
    st.dataframe(
        det.sort_values("Balance", ascending=False)[
            ["CustomerId", "Surname", "Geography", "Age", "Tenure", "NumOfProducts", "Balance",
             "EstimatedSalary", "Balance/Salary", "Exited"]
        ].head(500),
        hide_index=True, use_container_width=True)
    st.download_button("Download flagged list (CSV)", det.to_csv(index=False).encode(),
                       "flagged_customers.csv", "text/csv")
    st.plotly_chart(px.scatter(data.sample(min(len(data), 3000), random_state=1),
                               x="EstimatedSalary", y="Balance", color="Status",
                               title="Balance vs salary by engagement status (sample)", opacity=0.5),
                    use_container_width=True)

# ---------------------------------------------------------------- tab 4
with tabs[3]:
    st.subheader("Relationship Strength Index (RSI)")
    st.caption("RSI = 100 x product-tier retention x (0.5 + 0.5 x active flag). "
               "Weights are calibrated on this dataset, so treat it as an in-sample score.")
    f["RSI band"] = pd.cut(f.RSI, [-1, 20, 45, 70, 101], labels=["0-20", "20-45", "45-70", "70-100"])
    rb = f.groupby("RSI band", observed=True).Exited.agg(["size", rate]).reset_index()
    rb.columns = ["RSI band", "Customers", "Churn %"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(rb, x="RSI band", y="Churn %", text="Churn %",
                           title="Churn by RSI band").update_traces(texttemplate="%{text:.1f}%"),
                    use_container_width=True)
    c2.plotly_chart(px.histogram(f, x="RSI", nbins=30, title="RSI distribution"), use_container_width=True)

    st.subheader("Sticky customers")
    sticky = f[(f.IsActiveMember == 1) & (f.NumOfProducts == 2)]
    s = st.columns(3)
    s[0].metric("Sticky customers (active + 2 products)", f"{len(sticky):,}")
    s[1].metric("Share of filtered base", f"{len(sticky) / len(f) * 100:.1f}%")
    s[2].metric("Sticky churn rate", f"{rate(sticky.Exited):.2f}%" if len(sticky) else "n/a")

    st.subheader("Churn stability across engagement tiers")
    ba = f.assign(AgeBand=pd.cut(f.Age, [17, 30, 40, 50, 60, 100],
                                 labels=["18-30", "31-40", "41-50", "51-60", "60+"]))
    ab = ba.groupby(["AgeBand", "Status"], observed=True).Exited.apply(rate).reset_index(name="Churn %")
    st.plotly_chart(px.bar(ab, x="AgeBand", y="Churn %", color="Status", barmode="group",
                           title="Churn by age band and engagement"), use_container_width=True)

# ---------------------------------------------------------------- tab 5
with tabs[4]:
    st.subheader("Validation")
    st.write(f"Rows: {len(raw):,} | Columns: {raw.shape[1]}")
    st.write("Issues: " + (", ".join(issues) if issues else "none found."))
    st.write(f"Overall churn rate (full dataset): {rate(raw.Exited):.2f}%")
    st.dataframe(raw.describe().T.round(2), use_container_width=True)
    st.caption("Findings show association in a single snapshot, not causation.")
