import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# Shared colours used across all charts
BLUE   = "#4C6EF5"
RED    = "#E03131"
GREEN  = "#2F9E44"
ORANGE = "#F08C00"
PURPLE = "#7950F2"
GRAY   = "#868E96"
GRID   = "#EEEEEE"
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, GRAY, "#1098AD", "#D6336C"]

# Plotly's default font goes white when Streamlit is in dark mode, so we pin
# it to a dark grey here and set it as the global template default.
FONT_COLOR = "#1F2937"
pio.templates["cfpb"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial", color=FONT_COLOR),
        title=dict(font=dict(color=FONT_COLOR)),
        xaxis=dict(tickfont=dict(color=FONT_COLOR), title=dict(font=dict(color=FONT_COLOR))),
        yaxis=dict(tickfont=dict(color=FONT_COLOR), title=dict(font=dict(color=FONT_COLOR))),
        legend=dict(font=dict(color=FONT_COLOR)),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
)
pio.templates.default = "cfpb"

st.set_page_config(
    page_title="CFPB Consumer Complaints",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load once and cache — the parquet is ~130 MB so we don't want to re-read on every interaction
@st.cache_data
def load_data() -> pl.DataFrame:
    return pl.read_parquet("dashboard_data/cfpb_complaints_clean.parquet")

df_all = load_data()

# Sidebar filters drive all four tabs
with st.sidebar:
    st.title("Filters")
    st.caption("Filters apply to all tabs.")

    year_min = int(df_all["year"].min())
    year_max = int(df_all["year"].max())
    year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))

    all_products = sorted(df_all["Product"].unique().to_list())
    sel_products = st.multiselect("Product", all_products)

    all_states = sorted(df_all["State"].drop_nulls().unique().to_list())
    sel_states = st.multiselect("State", all_states)

    st.divider()
    st.caption("2019 data covers Jan–May only and is not directly comparable to full-year totals.")

    with st.expander("Data caveats"):
        null_state_pct = df_all["State"].is_null().mean() * 100
        st.markdown(f"""
**Known limitations in this dataset:**

- **Complaints are self-reported**, not a random sample — awareness and channel accessibility skew who files.
- **`Consumer disputed?` was discontinued in 2017.** Any dispute rate shown for post-2017 data reflects missing data, not the absence of disputes.
- **2019 is a partial year** (January–May only); complaint totals cannot be compared to other full years.
- **Company names are inconsistent** — the same institution may appear under multiple name variants (e.g. "Equifax, Inc." vs "Equifax Information Services LLC"), inflating apparent competitor diversity.
- **State field is missing for {null_state_pct:.1f}% of records** and those rows are excluded from the geographic map.
- This dataset covers complaints *received* by the CFPB — downstream resolution quality and consumer outcomes beyond CFPB involvement are not captured.
        """)

# Apply whatever the user selected — empty multiselect means "show all"
df = df_all.filter(
    (pl.col("year") >= year_range[0]) & (pl.col("year") <= year_range[1])
)
if sel_products:
    df = df.filter(pl.col("Product").is_in(sel_products))
if sel_states:
    df = df.filter(pl.col("State").is_in(sel_states))

# Guard against a filter combination that returns nothing
if df.is_empty():
    st.warning("No data matches the current filters. Try adjusting the year range, product, or state selection.")
    st.stop()

# --- KPI delta helpers ---
# Compare the last two complete years in the filtered range (exclude partial 2019)
complete_years = sorted([y for y in df["year"].unique().to_list() if y < 2019])
if len(complete_years) >= 2:
    last_yr, prev_yr = complete_years[-1], complete_years[-2]
    df_last = df.filter(pl.col("year") == last_yr)
    df_prev = df.filter(pl.col("year") == prev_yr)
    show_delta = True
else:
    show_delta = False

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview",
    "📦 Product Analysis",
    "🏦 Company Analysis",
    "📬 Channels & Response",
])

# --- Overview tab ---
with tab1:
    st.subheader("Overview")

    total         = df.shape[0]
    timely_pct    = (df["Timely response?"] == "Yes").mean() * 100
    avg_days      = df["response_days"].drop_nulls().mean()
    narrative_pct = df["Consumer complaint narrative"].is_not_null().mean() * 100

    # Deltas vs the previous full year so you can see whether things are improving
    if show_delta:
        d_total     = df_last.shape[0] - df_prev.shape[0]
        d_timely    = (df_last["Timely response?"] == "Yes").mean() * 100 - (df_prev["Timely response?"] == "Yes").mean() * 100
        d_days      = df_last["response_days"].drop_nulls().mean() - df_prev["response_days"].drop_nulls().mean()
        d_narrative = df_last["Consumer complaint narrative"].is_not_null().mean() * 100 - df_prev["Consumer complaint narrative"].is_not_null().mean() * 100
    else:
        d_total = d_timely = d_days = d_narrative = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Complaints",     f"{total:,}",
              delta=f"{d_total:+,} vs {prev_yr}" if show_delta else None)
    c2.metric("Timely Response Rate", f"{timely_pct:.1f}%",
              delta=f"{d_timely:+.1f}pp vs {prev_yr}" if show_delta else None)
    # More response days is worse, so flip the delta colour
    c3.metric("Avg Response Days",    f"{avg_days:.1f}",
              delta=f"{d_days:+.1f}d vs {prev_yr}" if show_delta else None,
              delta_color="inverse")
    c4.metric("With Narrative",       f"{narrative_pct:.1f}%",
              delta=f"{d_narrative:+.1f}pp vs {prev_yr}" if show_delta else None)

    # --- Key Findings: synthesise what the numbers actually mean ---
    st.divider()
    st.subheader("Key Findings")

    top_prod = (
        df.group_by("Product").agg(pl.len().alias("n"))
        .sort("n", descending=True).row(0, named=True)
    )
    top_prod_pct = top_prod["n"] / max(df.shape[0], 1) * 100

    top_issue = (
        df.group_by("Issue").agg(pl.len().alias("n"))
        .sort("n", descending=True).row(0, named=True)
    )

    worst_co_df = (
        df.group_by("Company")
        .agg([pl.len().alias("n"), (pl.col("Timely response?") == "No").mean().alias("late_pct")])
        .filter(pl.col("n") >= 5000)
        .sort("late_pct", descending=True)
    )

    kf1, kf2, kf3, kf4 = st.columns(4)

    with kf1:
        st.info(
            f"**Dominant pain point**\n\n"
            f"**{top_prod['Product']}** is the top category at **{top_prod_pct:.0f}%** of all complaints. "
            f"The single most-reported issue is *\"{top_issue['Issue'][:55]}\"* "
            f"({top_issue['n']:,} complaints)."
        )

    with kf2:
        if show_delta:
            vol_chg = (df_last.shape[0] - df_prev.shape[0]) / max(df_prev.shape[0], 1) * 100
            direction = "grew" if vol_chg > 0 else "fell"
            st.info(
                f"**Volume trend**\n\n"
                f"Complaint volume **{direction} {abs(vol_chg):.0f}%** from {prev_yr} to {last_yr} "
                f"({df_prev.shape[0]:,} → {df_last.shape[0]:,}). "
            )
        else:
            st.info(
                f"**Total volume**\n\n"
                f"**{df.shape[0]:,}** complaints in scope. Broaden the year filter to see year-on-year trends."
            )

    with kf3:
        if show_delta:
            tr_last = (df_last["Timely response?"] == "Yes").mean() * 100
            tr_prev = (df_prev["Timely response?"] == "Yes").mean() * 100
            tr_delta = tr_last - tr_prev
            tr_dir = "improved" if tr_delta > 0 else "declined"
            icon = "A positive signal." if tr_delta > 0 else "Warrants investigation."
            st.info(
                f"**Response quality**\n\n"
                f"Timely response rate **{tr_dir} {abs(tr_delta):.1f} pp** from {prev_yr} to {last_yr}, "
                f"reaching **{tr_last:.1f}%**. {icon}"
            )
        else:
            st.info(
                f"**Response quality**\n\n"
                f"Timely response rate across the selected period: **{timely_pct:.1f}%**. "
                f"Avg resolution: **{avg_days:.1f} days**."
            )

    with kf4:
        if not worst_co_df.is_empty():
            wc = worst_co_df.row(0, named=True)
            st.warning(
                f"**Operational risk**\n\n"
                f"**{wc['Company'][:45]}** has the highest late-response rate among large filers: "
                f"**{wc['late_pct']*100:.0f}%** across {wc['n']:,} complaints. "
                f"This is an outlier worth escalating."
            )
        else:
            st.info(
                "**Operational risk**\n\n"
                "No single company reaches 5,000 complaints in the current filter — "
                "broaden the scope to identify concentration risk."
            )

    st.divider()
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Year-on-year volume with growth % annotations on each point
        yearly = df.group_by("year").agg(pl.len().alias("complaints")).sort("year")
        years  = yearly["year"].to_list()
        counts = yearly["complaints"].to_list()

        fig = go.Figure()

        # Solid filled area covers all complete years; 2019 gets a dashed bridge
        if 2019 in years:
            cut = years.index(2019)
            solid_x, solid_y = years[:cut], counts[:cut]
        else:
            cut = len(years)
            solid_x, solid_y = years, counts

        fig.add_trace(go.Scatter(
            x=solid_x, y=solid_y,
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=8, color=BLUE),
            fill="tozeroy", fillcolor="rgba(76, 110, 245, 0.08)",
            showlegend=False,
        ))

        # YoY growth percentage above each point (complete years only)
        for i in range(1, len(solid_x)):
            pct = (solid_y[i] - solid_y[i - 1]) / solid_y[i - 1] * 100
            color = GREEN if pct >= 0 else RED
            fig.add_annotation(
                x=solid_x[i], y=solid_y[i],
                text=f"{pct:+.0f}%",
                showarrow=False, yshift=14,
                font=dict(size=9, color=color),
            )

        # Dashed bridge to partial 2019 if it's in range
        if 2019 in years:
            actual_2019   = counts[cut]
            projected_2019 = round(actual_2019 / 5 * 12)  # Jan–May → annualized

            fig.add_trace(go.Scatter(
                x=[years[cut - 1], years[cut]], y=[counts[cut - 1], counts[cut]],
                mode="lines+markers",
                line=dict(color=GRAY, width=2, dash="dot"),
                marker=dict(size=8, color=GRAY, symbol="circle-open"),
                showlegend=False,
            ))
            # Faint diamond showing where the annualized pace would land
            fig.add_trace(go.Scatter(
                x=[2019], y=[projected_2019],
                mode="markers",
                marker=dict(size=10, color=GRAY, symbol="diamond-open", opacity=0.6),
                showlegend=False,
            ))
            fig.add_annotation(
                x=2019, y=actual_2019,
                text=f"Jan–May: {actual_2019:,}<br>~{projected_2019:,} annualized ◇",
                showarrow=True, arrowhead=2, arrowcolor=GRAY,
                ax=60, ay=-45,
                font=dict(color=GRAY, size=10),
                align="left",
            )

        fig.update_layout(
            title="Complaint Volume by Year",
            xaxis=dict(dtick=1, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0, r=0, t=40, b=0),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Dotted line: 2019 data covers January–May only and is not comparable to full-year figures. "
            "◇ Diamond marker shows the annualized pace at the May cut-off."
        )

    with col_right:
        # Complement to the trend line — shows which products dominate overall
        prod_counts = (
            df.group_by("Product")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(10)
        )
        labels = [p[:30] for p in prod_counts["Product"].to_list()[::-1]]
        values = prod_counts["n"].to_list()[::-1]

        fig2 = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=BLUE,
        ))
        fig2.update_layout(
            title="Top Products",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=0, r=0, t=40, b=0),
            height=340,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # State choropleth — shows the geographic distribution of complaints
    st.divider()
    state_counts = (
        df.drop_nulls(subset=["State"])
        .group_by("State")
        .agg(pl.len().alias("complaints"))
    )
    fig_map = go.Figure(go.Choropleth(
        locations=state_counts["State"].to_list(),
        z=state_counts["complaints"].to_list(),
        locationmode="USA-states",
        colorscale="Blues",
        colorbar=dict(title="Complaints", tickfont=dict(color=FONT_COLOR)),
    ))
    fig_map.update_layout(
        title="Complaints by State",
        geo=dict(scope="usa", bgcolor="white", lakecolor="white", landcolor="#F9FAFB"),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        height=380,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # --- What to Act On: translate findings into prioritised decisions ---
    st.divider()
    st.subheader("What to Act On")
    st.caption("Prioritised recommendations based on the data in scope. Adjust filters to refocus on a specific product or state.")

    act1, act2, act3 = st.columns(3)

    with act1:
        top3_issues = (
            df.group_by("Issue").agg(pl.len().alias("n"))
            .sort("n", descending=True).head(3)["Issue"].to_list()
        )
        bullets = "\n".join(f"- {i[:55]}" for i in top3_issues)
        st.error(
            f"**Priority 1 — Reduce top-category complaint volume**\n\n"
            f"**{top_prod['Product']}** generates {top_prod_pct:.0f}% of all complaints. "
            f"The three issues driving it are:\n{bullets}\n\n"
            f"These are the highest-leverage issues to address with policy or process changes."
        )

    with act2:
        if not worst_co_df.is_empty():
            wc = worst_co_df.row(0, named=True)
            top3_late = worst_co_df.head(3)
            bullets = "\n".join(
                f"- {r['Company'][:40]} ({r['late_pct']*100:.0f}% late)"
                for r in top3_late.iter_rows(named=True)
            )
            st.warning(
                f"**Priority 2 — Enforce response-time standards**\n\n"
                f"Several large companies consistently miss the timely-response threshold:\n{bullets}\n\n"
                f"Targeted enforcement or supervisory letters to these firms would move the industry-wide metric."
            )
        else:
            st.warning(
                "**Priority 2 — Enforce response-time standards**\n\n"
                "Apply a broader filter to identify companies with chronic late-response patterns. "
                "A timely-response rate below 95% at any large filer is a supervisory flag."
            )

    with act3:
        st.info(
            "**Priority 3 — Close the dispute data gap**\n\n"
            "The *Consumer disputed?* field was discontinued in 2017, leaving a blind spot in quality measurement. "
            "Without it, it's impossible to tell whether company responses are actually resolving complaints "
            "or just closing them. **Re-instating or replacing this field** is necessary for meaningful outcome tracking.\n\n"
            "Until then, use *response type* distribution (Channels tab) as a proxy, "
            "a high share of 'Closed without relief' is a warning sign."
        )

# --- Product Analysis tab ---
with tab2:
    st.subheader("Product Analysis")

    # The dispute field was deprecated by the CFPB in 2017, so products that
    # appear mostly in post-2017 data will show artificially low dispute rates.
    st.info(
        "**Dispute rate caveat:** The *Consumer disputed?* field was discontinued in 2017. "
        "Products with complaints that fall mostly after 2017 will show a dispute rate near 0% "
        "— this reflects missing data, not the absence of disputes.",
        icon="ℹ️",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        # dispute_pct uses fill_null(0) because the field was deprecated in 2017
        # so post-2017 records will all be null — we treat those as "no dispute"
        risk = (
            df.group_by("Product")
            .agg([
                (pl.col("Timely response?") == "No").mean().alias("late_pct"),
                (pl.col("Consumer disputed?") == "Yes").mean().fill_null(0).alias("dispute_pct"),
            ])
            .sort("dispute_pct", descending=True)
        )
        products  = [p[:30] for p in risk["Product"].to_list()]
        late_vals = [round(v * 100, 1) for v in risk["late_pct"].to_list()]
        disp_vals = [round(v * 100, 1) for v in risk["dispute_pct"].to_list()]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Late response %",    x=products, y=late_vals, marker_color=BLUE))
        fig.add_trace(go.Bar(name="Consumer dispute %", x=products, y=disp_vals, marker_color=RED))
        fig.update_layout(
            barmode="group",
            title="Late Response & Dispute Rate by Product",
            yaxis_title="Rate (%)",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.08),
            xaxis_tickangle=-35,
            margin=dict(l=0, r=0, t=60, b=100),
            height=440,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Top issues across all products — useful for spotting systemic themes
        issues = (
            df.group_by("Issue")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(15)
        )
        labels = [i[:45] for i in issues["Issue"].to_list()[::-1]]
        vals   = issues["n"].to_list()[::-1]

        fig2 = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=BLUE,
        ))
        fig2.update_layout(
            title="Top 15 Issues",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
            yaxis=dict(tickfont=dict(size=10), automargin=True),
            margin=dict(l=0, r=0, t=40, b=0),
            height=440,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Response quality over time — timely rate by year, broken out by product when filtered
    st.divider()
    if sel_products:
        st.markdown("**Timely Response Rate Over Time — Selected Products**")
        rq_data = (
            df.filter(pl.col("year") < 2019)
            .group_by(["Product", "year"])
            .agg(((pl.col("Timely response?") == "Yes").mean() * 100).alias("timely_pct"))
            .sort("year")
        )
        fig_rq = go.Figure()
        for i, prod in enumerate(sel_products):
            sub = rq_data.filter(pl.col("Product") == prod)
            fig_rq.add_trace(go.Scatter(
                x=sub["year"].to_list(),
                y=[round(v, 1) for v in sub["timely_pct"].to_list()],
                name=prod[:40],
                mode="lines+markers",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                marker=dict(size=6),
            ))
    else:
        st.markdown("**Timely Response Rate Over Time — All Products**")
        rq_data = (
            df.filter(pl.col("year") < 2019)
            .group_by("year")
            .agg(((pl.col("Timely response?") == "Yes").mean() * 100).alias("timely_pct"))
            .sort("year")
        )
        fig_rq = go.Figure(go.Scatter(
            x=rq_data["year"].to_list(),
            y=[round(v, 1) for v in rq_data["timely_pct"].to_list()],
            mode="lines+markers",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=8, color=GREEN),
            fill="tozeroy", fillcolor="rgba(47, 158, 68, 0.07)",
            showlegend=False,
        ))

    fig_rq.update_layout(
        xaxis=dict(dtick=1, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="Timely response (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=0, r=0, t=20, b=0),
        height=300,
    )
    st.plotly_chart(fig_rq, use_container_width=True)
    st.caption("2019 excluded (partial year). A declining trend indicates deteriorating response standards across the industry.")

    # Only show the per-product trend lines when no product filter is active —
    # filtering to one product makes this chart pointless
    if not sel_products:
        st.divider()
        st.markdown("**Complaint Volume Over Time — Top 6 Products**")

        top6 = (
            df.group_by("Product")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(6)["Product"]
            .to_list()
        )
        prod_yearly = (
            df.filter(pl.col("Product").is_in(top6))
            .group_by(["Product", "year"])
            .agg(pl.len().alias("complaints"))
            .sort("year")
        )

        fig3 = go.Figure()
        for i, prod in enumerate(top6):
            sub = prod_yearly.filter(pl.col("Product") == prod)
            fig3.add_trace(go.Scatter(
                x=sub["year"].to_list(),
                y=sub["complaints"].to_list(),
                name=prod[:40],
                mode="lines+markers",
                line=dict(color=PALETTE[i], width=2),
                marker=dict(size=6),
            ))
        fig3.update_layout(
            xaxis=dict(dtick=1, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=0, r=0, t=20, b=0),
            height=360,
        )
        st.plotly_chart(fig3, use_container_width=True)


# --- Company Analysis tab ---
with tab3:
    st.subheader("Company Analysis")

    # Search box so you can look up a specific company without scrolling the full list
    search_term = st.text_input("Search company", placeholder="e.g. Wells Fargo")
    if search_term:
        company_df = df.filter(
            pl.col("Company").str.to_lowercase().str.contains(search_term.lower())
        )
        if company_df.is_empty():
            st.warning(f"No companies found matching '{search_term}'.")
            company_df = df
    else:
        company_df = df

    col_left, col_right = st.columns([1, 1])

    with col_left:
        top_n = st.slider("Show top N companies", 10, 30, 20, key="top_n")
        top_companies = (
            company_df.group_by("Company")
            .agg(pl.len().alias("complaints"))
            .sort("complaints", descending=True)
            .head(top_n)
        )
        labels = [c[:40] for c in top_companies["Company"].to_list()[::-1]]
        vals   = top_companies["complaints"].to_list()[::-1]

        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=BLUE,
            text=[f"{v:,}" for v in vals],
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig.update_layout(
            title=f"Top {top_n} Companies by Complaint Volume",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
            yaxis=dict(tickfont=dict(size=9), automargin=True),
            margin=dict(l=0, r=60, t=40, b=0),
            height=max(400, top_n * 22),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Filter to companies with at least 1,000 complaints so the rates are meaningful
        company_risk = (
            company_df.group_by("Company")
            .agg([
                pl.len().alias("total"),
                (pl.col("Timely response?") == "No").mean().alias("late_pct"),
                (pl.col("Consumer disputed?") == "Yes").mean().fill_null(0).alias("dispute_pct"),
            ])
            .filter(pl.col("total") >= 1000)
            .sort("late_pct", descending=True)
            .head(20)
        )
        companies = [c[:35] for c in company_risk["Company"].to_list()]
        late_vals = [round(v * 100, 1) for v in company_risk["late_pct"].to_list()]
        disp_vals = [round(v * 100, 1) for v in company_risk["dispute_pct"].to_list()]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Late response %",    x=late_vals, y=companies, orientation="h", marker_color=BLUE))
        fig2.add_trace(go.Bar(name="Consumer dispute %", x=disp_vals, y=companies, orientation="h", marker_color=RED))
        fig2.update_layout(
            barmode="group",
            title="Response Quality — Top 20 by Late Rate (min. 1,000 complaints)",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.08, x=0, xanchor="left"),
            xaxis=dict(showgrid=True, gridcolor=GRID, title="Rate (%)"),
            # Pin category order so both traces align — without this Plotly can
            # infer different orderings per-trace and shift individual bars.
            yaxis=dict(
                tickfont=dict(size=9), automargin=True,
                categoryorder="array", categoryarray=companies,
            ),
            margin=dict(l=0, r=0, t=40, b=60),
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)


# --- Channels & Response tab ---
with tab4:
    st.subheader("Channels & Company Response")

    col1, col2, col3 = st.columns(3)

    with col1:
        channels = df["Submitted via"].value_counts(sort=True)
        fig = go.Figure(go.Pie(
            labels=channels["Submitted via"].to_list(),
            values=channels["count"].to_list(),
            hole=0.45,
            marker_colors=PALETTE,
            # Keep labels inside or auto-placed; automargin handles overflow
            textposition="auto",
        ))
        fig.update_layout(
            title="Submission Channel",
            # Extra horizontal margin so labels don't get clipped by the column edge
            margin=dict(l=30, r=30, t=40, b=20),
            height=340,
            autosize=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        responses = (
            df["Company response to consumer"]
            .fill_null("Unknown")
            .value_counts(sort=True)
        )
        fig2 = go.Figure(go.Bar(
            x=responses["count"].to_list(),
            y=[r[:35] for r in responses["Company response to consumer"].to_list()],
            orientation="h",
            marker_color=BLUE,
        ))
        fig2.update_layout(
            title="Company Response Type",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
            yaxis=dict(tickfont=dict(size=10), automargin=True),
            margin=dict(l=0, r=0, t=40, b=0),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        # Colour the slices explicitly so green always means on-time
        timely = df["Timely response?"].value_counts()
        label_list = timely["Timely response?"].to_list()
        color_list = [GREEN if v == "Yes" else RED for v in label_list]

        fig3 = go.Figure(go.Pie(
            labels=label_list,
            values=timely["count"].to_list(),
            hole=0.45,
            marker_colors=color_list,
        ))
        fig3.update_layout(
            title="Timely Response",
            margin=dict(l=0, r=0, t=40, b=0),
            height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Channel mix over time — shows the structural shift from phone/postal to web
    st.divider()
    st.markdown("**Submission Channel Mix Over Time**")
    channel_trend = (
        df.filter(pl.col("year") < 2019)
        .group_by(["year", "Submitted via"])
        .agg(pl.len().alias("n"))
        .sort("year")
    )
    top_channels = (
        df["Submitted via"].value_counts(sort=True).head(6)["Submitted via"].to_list()
    )
    fig_ct = go.Figure()
    for i, ch in enumerate(top_channels):
        sub = channel_trend.filter(pl.col("Submitted via") == ch)
        fig_ct.add_trace(go.Scatter(
            x=sub["year"].to_list(),
            y=sub["n"].to_list(),
            name=ch,
            mode="lines+markers",
            stackgroup="one",  # stacked area so shares are visible
            line=dict(color=PALETTE[i % len(PALETTE)], width=1),
            marker=dict(size=4),
        ))
    fig_ct.update_layout(
        xaxis=dict(dtick=1, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="Complaints"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
    )
    st.plotly_chart(fig_ct, use_container_width=True)
    st.caption(
        "Web submissions have grown to dominate the channel mix. "
        "A shift away from phone/postal indicates consumers are becoming more digitally engaged — "
        "and that web-channel complaint resolution processes are now the most critical to scale."
    )

    st.divider()

    if not sel_products:
        # Showing all 10+ products at once makes the box plot hard to read.
        # Prompt the user to filter so the chart is actually useful.
        st.info(
            "Select one or more products in the sidebar to see the response time distribution.",
            icon="ℹ️",
        )
    else:
        # Box plot showing the full spread of response times, not just the median.
        # Capped at 30 days to keep the x-axis readable — extreme outliers exist but
        # are not meaningful for operational comparisons.
        box_stats = (
            df.filter(
                pl.col("response_days").is_not_null() &
                (pl.col("response_days") >= 0) &
                (pl.col("response_days") <= 30)
            )
            .group_by("Product")
            .agg([
                pl.col("response_days").quantile(0.05).alias("p5"),
                pl.col("response_days").quantile(0.25).alias("q1"),
                pl.col("response_days").median().alias("median"),
                pl.col("response_days").mean().alias("mean"),
                pl.col("response_days").quantile(0.75).alias("q3"),
                pl.col("response_days").quantile(0.95).alias("p95"),
            ])
            .sort("median", descending=True)
        )

        fig4 = go.Figure()
        for row in box_stats.iter_rows(named=True):
            fig4.add_trace(go.Box(
                name=row["Product"][:35],
                lowerfence=[row["p5"]],
                q1=[row["q1"]],
                median=[row["median"]],
                mean=[row["mean"]],
                q3=[row["q3"]],
                upperfence=[row["p95"]],
                orientation="h",
                showlegend=False,
                marker_color=BLUE,
                line_color=BLUE,
                fillcolor="rgba(76, 110, 245, 0.15)",
            ))

        n_products = len(box_stats)
        fig4.update_layout(
            title="Response Time Distribution by Product (days, capped at 30, whiskers = 5th–95th percentile)",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor=GRID, title="Days", zeroline=False),
            yaxis=dict(automargin=True),
            margin=dict(l=0, r=20, t=50, b=0),
            height=max(280, n_products * 60 + 100),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Box = interquartile range (25th–75th percentile). Line = median. Cross = mean. Whiskers = 5th–95th percentile.")
