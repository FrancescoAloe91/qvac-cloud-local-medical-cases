"""Plotly charts for the benchmark dashboard."""

from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.i18n import t
from lib.tiers import MODEL_CONFIG


def _chart_labels():
    """Lazy import — breaks circular import charts ↔ metrics at module load."""
    from lib.metrics import TABLE_MODEL_SHORT, URGENCY_META, _L

    return TABLE_MODEL_SHORT, URGENCY_META, _L


def _short_chart_label(key: str, tier_labels: Optional[dict] = None) -> str:
    from lib.cloud_tiers import short_chart_label

    return short_chart_label(key, tier_labels)


MODEL_COLORS = {
    "chatgpt": "#10a37f",
    "claude": "#d97706",
    "gemini": "#8ab4f8",
    "qvac": "#00d09c",
}

MODEL_CONFIG_NAME = {k: v["name"] for k, v in MODEL_CONFIG.items()}

FILL_COLORS = {
    "chatgpt": "rgba(16,163,127,0.18)",
    "claude": "rgba(217,119,6,0.18)",
    "gemini": "rgba(138,180,248,0.18)",
    "qvac": "rgba(0,208,156,0.22)",
}

CHART_BG = "#0e1117"
PLOT_BG = "#161b26"
GRID = "#2a3142"


def _chart_model_label(key: str, full_name: str, tier_labels: Optional[dict] = None) -> str:
    if tier_labels is not None:
        return _short_chart_label(key, tier_labels)
    table_short, _, _ = _chart_labels()
    return table_short.get(key, full_name)


def _chart_bar_tick(key: str, display_name: str) -> str:
    """Short Y-axis labels so bars stay visible in narrow columns; version lives in the table."""
    table_short, _, _ = _chart_labels()
    return table_short.get(key, display_name)


def _base_layout(title: str, height: int = 500) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color="#fafafa")),
        template="plotly_dark",
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        font=dict(color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )


def fig_radar(
    ranking_df,
    use_gold: bool,
    lang: str = "en",
    sem_available: bool = False,
    tier_labels: Optional[dict] = None,
) -> go.Figure:
    _ = tier_labels
    _, __, _L = _chart_labels()
    L = _L(lang)
    if use_gold:
        col_keys = [L["rel_short"], L["acc_cons_short"]]
        if sem_available:
            col_keys.append(L["sem_short"])
        col_keys += [
            L["acc_clin_short"],
            L["ddx_short"],
            L["score_cons_rescaled"],
            L["score_clin_short"],
        ]
        categories = col_keys[:]
    else:
        col_keys = [
            L["acc_cons_short"],
            L["sem_short"],
            L["rel_short"],
            L["urg_dim_short"],
            L["score_cons_rescaled"],
        ]
        categories = col_keys[:]

    fig = go.Figure()
    for _, row in ranking_df.iterrows():
        key = row["key"]
        color = MODEL_COLORS.get(key, "#94a3b8")
        values = [row[c] if row[c] is not None else 0 for c in col_keys]
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                name=row[L["model"]],
                line=dict(color=color, width=2.5),
                fill="toself",
                fillcolor=FILL_COLORS.get(key, "rgba(148,163,184,0.15)"),
                marker=dict(size=7, color=color),
            )
        )

    fig.update_layout(
        **_base_layout(t("chart.radar_title", lang), height=440),
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=GRID,
                linecolor=GRID,
                tickfont=dict(size=10),
            ),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID),
        ),
    )
    return fig


def fig_ranking_bars(ranking_df, use_gold: bool, lang: str = "en", height: int = 320) -> go.Figure:
    """Single ranking chart — consensus (relative) or clinical (absolute) only."""
    return fig_consensus_ranking_bars(ranking_df, lang, height)


def fig_consensus_ranking_bars(
    ranking_df, lang: str = "en", height: int = 260, tier_labels: Optional[dict] = None
) -> go.Figure:
    """Consensus ranking: best model in the group = 100%, others scale down."""
    _ = tier_labels
    _, __, _L = _chart_labels()
    L = _L(lang)
    models = [
        _chart_bar_tick(row["key"], row[L["model"]])
        for _, row in ranking_df.iterrows()
    ]
    colors = [MODEL_COLORS.get(k, "#94a3b8") for k in ranking_df["key"]]
    fig = go.Figure(
        go.Bar(
            y=models,
            x=ranking_df[L["score_cons_rescaled"]],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#1e293b", width=1)),
            text=[
                f"#{int(r)} · {v}%"
                for r, v in zip(ranking_df[L["rank_consensus"]], ranking_df[L["score_cons_rescaled"]])
            ],
            textposition="outside",
        )
    )
    layout = _base_layout(t("chart.bars_consensus", lang), height=height)
    layout["margin"] = dict(l=40, r=72, t=60, b=30)
    layout["xaxis"] = dict(range=[0, 110], title=t("chart.score_pct", lang))
    fig.update_layout(**layout)
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_clinical_ranking_bars(
    ranking_df, lang: str = "en", height: int = 260, tier_labels: Optional[dict] = None
) -> go.Figure:
    """Clinical ranking vs. confirmed diagnosis — absolute % (100% = perfect match with reference)."""
    _ = tier_labels
    _, __, _L = _chart_labels()
    L = _L(lang)
    score_col = L["score_clin_short"]
    models = [
        _chart_bar_tick(row["key"], row[L["model"]])
        for _, row in ranking_df.iterrows()
    ]
    fig = go.Figure(
        go.Bar(
            y=models,
            x=ranking_df[score_col],
            orientation="h",
            marker=dict(color="#f5c518", line=dict(color="#1e293b", width=1)),
            text=[
                f"#{int(r)} · {v}%"
                for r, v in zip(ranking_df[L["rank_clinical"]], ranking_df[score_col])
            ],
            textposition="outside",
        )
    )
    layout = _base_layout(t("chart.bars_clinical", lang), height=height)
    layout["margin"] = dict(l=40, r=72, t=60, b=30)
    layout["xaxis"] = dict(range=[0, 110], title=t("chart.score_pct", lang))
    fig.update_layout(**layout)
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_dimensions_grouped(ranking_df, use_gold: bool, lang: str = "en", sem_available: bool = False) -> go.Figure:
    _, __, _L = _chart_labels()
    L = _L(lang)
    models = ranking_df[L["model"]].tolist()
    fig = go.Figure()

    dims = [
        (L["acc_cons_short"], "#3b82f6"),
        (L["sem_short"], "#a855f7"),
        (L["rel_short"], "#6366f1"),
        (L["urg_dim_short"], "#f97316"),
    ]
    if use_gold:
        dims += [
            (L["acc_clin_short"], "#f5c518"),
            (L["ddx_short"], "#eab308"),
        ]

    for dim_name, color in dims:
        fig.add_bar(name=dim_name, x=models, y=ranking_df[dim_name].fillna(0), marker_color=color)

    fig.update_layout(
        **_base_layout(t("chart.dims_title", lang), height=360),
        barmode="group",
        yaxis=dict(range=[0, 105], title=t("chart.score_pct", lang)),
        xaxis=dict(title=""),
    )
    return fig


def fig_privacy_gauges(
    ranking_df, lang: str = "en", height: int = 300, tier_labels: Optional[dict] = None
) -> go.Figure:
    """Gauge per modello: 0% cloud → 100% on-device — tutti sulla stessa riga."""
    _ = tier_labels
    _, __, _L = _chart_labels()
    L = _L(lang)
    rows = list(ranking_df.iterrows())
    n = max(len(rows), 1)
    cols = n
    rows_n = 1
    fig = go.Figure()
    annotations = []

    x_gap, y_gap = 0.04, 0.12
    cell_w = (1.0 - x_gap * (cols + 1)) / cols
    cell_h = 0.72

    for i, (_, row) in enumerate(rows):
        key = row["key"]
        color = MODEL_COLORS.get(key, "#94a3b8")
        label = _chart_bar_tick(key, row[L["model"]])
        x0 = x_gap + i * (cell_w + x_gap)
        x1 = x0 + cell_w
        y0 = y_gap
        y1 = y0 + cell_h
        gauge_y0 = y0 + cell_h * 0.2
        gauge_y1 = y0 + cell_h * 0.95
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=row[L["privacy"]],
                number=dict(
                    suffix="%",
                    font=dict(size=16, color=color),
                    valueformat=".0f",
                ),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=GRID, tickfont=dict(size=7), nticks=3),
                    bar=dict(color=color, thickness=0.42),
                    bgcolor=PLOT_BG,
                    borderwidth=0,
                    steps=[
                        {"range": [0, 40], "color": "rgba(239,68,68,0.18)"},
                        {"range": [40, 75], "color": "rgba(245,197,24,0.14)"},
                        {"range": [75, 100], "color": "rgba(0,208,156,0.14)"},
                    ],
                ),
                domain={"x": [x0, x1], "y": [gauge_y0, gauge_y1]},
            )
        )
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=y0 - 0.02,
                xref="paper",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=10, color="#e2e8f0"),
                xanchor="center",
                yanchor="top",
            )
        )

    layout = _base_layout(t("chart.privacy_title", lang), height=max(height, 220))
    layout["margin"] = dict(l=8, r=8, t=64, b=28)
    layout["annotations"] = annotations
    fig.update_layout(**layout)
    return fig


def fig_concordance_heatmap(
    matrix_keyed: dict, model_keys: list, lang: str = "en", tier_labels: Optional[dict] = None
) -> go.Figure:
    _ = tier_labels
    """Heatmap di concordanza diagnostica pairwise tra tutti i modelli attivi."""
    names = [MODEL_CONFIG_NAME.get(k, k) for k in model_keys]
    z = []
    for a in model_keys:
        row = []
        for b in model_keys:
            if a == b:
                row.append(100.0)
            else:
                row.append(matrix_keyed.get((a, b), 0.0))
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=names,
            y=names,
            colorscale=[
                [0.0, "#1e293b"],
                [0.4, "#3b82f6"],
                [0.75, "#00d09c"],
                [1.0, "#f5c518"],
            ],
            zmin=0,
            zmax=100,
            text=[[f"{v:.0f}%" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=13),
            hovertemplate="%{y} ↔ %{x}: %{z:.1f}%<extra></extra>",
            colorbar=dict(title=t("chart.score_pct", lang), ticksuffix="%"),
        )
    )
    layout = _base_layout(t("chart.heatmap_title", lang), height=380)
    layout["xaxis"] = dict(side="top")
    layout["margin"] = dict(l=40, r=40, t=90, b=20)
    fig.update_layout(**layout)
    return fig


def fig_keyword_bar(keyword_counts: dict, total_models: int, lang: str = "en") -> go.Figure:
    """Barre delle keyword cliniche di consenso, ordinate per frequenza tra modelli."""
    if not keyword_counts:
        return go.Figure()
    items = sorted(keyword_counts.items(), key=lambda kv: kv[1], reverse=True)[:14]
    items.reverse()
    words = [w for w, _ in items]
    counts = [c for _, c in items]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=words,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale=[[0, "#3b82f6"], [1, "#00d09c"]],
                line=dict(color="#1e293b", width=1),
            ),
            text=[f"{c}/{total_models}" for c in counts],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout(t("chart.keywords_title", lang), height=380),
        xaxis=dict(title=t("chart.keywords_axis", lang), range=[0, total_models + 0.5]),
        yaxis=dict(title=""),
    )
    return fig


def fig_urgency_compare(urgency: dict, lang: str = "en", tier_labels: Optional[dict] = None) -> go.Figure:
    _ = tier_labels
    """Confronto del livello di urgenza/triage dichiarato da ciascun modello."""
    table_short, urgency_meta, _ = _chart_labels()
    keys, scores, colors, labels = [], [], [], []
    for key, u in urgency.items():
        meta = urgency_meta.get(u.get("label"), urgency_meta[None])
        keys.append(table_short.get(key, MODEL_CONFIG_NAME.get(key, key)))
        scores.append(u.get("score") or 0)
        colors.append(meta["color"])
        labels.append(t(meta["label_key"], lang))

    fig = go.Figure(
        go.Bar(
            x=keys,
            y=scores,
            marker=dict(color=colors, line=dict(color="#1e293b", width=1)),
            text=labels,
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout(t("chart.urgency_title", lang), height=320),
        yaxis=dict(range=[0, 115], title=t("chart.score_pct", lang)),
        xaxis=dict(title=""),
    )
    return fig


def fig_leaderboard(leaderboard_df, lang: str = "en", tier_labels: Optional[dict] = None) -> go.Figure:
    _ = tier_labels
    """Classifica cumulativa vittorie/score medio tra i round eseguiti in sessione."""
    _, __, _L = _chart_labels()
    L = _L(lang)
    if leaderboard_df is None or leaderboard_df.empty:
        return go.Figure()
    colors = [MODEL_COLORS.get(k, "#94a3b8") for k in leaderboard_df["key"]]
    wins_col = t("leaderboard.wins", lang)
    avg_col = t("leaderboard.avg_score", lang)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(wins_col, avg_col), horizontal_spacing=0.12)
    fig.add_trace(
        go.Bar(
            x=leaderboard_df[L["model"]],
            y=leaderboard_df[wins_col],
            marker_color=colors,
            showlegend=False,
            text=leaderboard_df[wins_col],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=leaderboard_df[L["model"]],
            y=leaderboard_df[avg_col],
            marker_color=colors,
            showlegend=False,
            text=[f"{v}%" for v in leaderboard_df[avg_col]],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    layout = _base_layout(t("chart.leaderboard_title", lang), height=340)
    fig.update_layout(**layout)
    fig.update_yaxes(range=[0, max(leaderboard_df[wins_col].max() + 1, 3)], row=1, col=1)
    fig.update_yaxes(range=[0, 110], row=1, col=2)
    return fig


def fig_judge_accuracy_bars(
    ranking_rows: list,
    *,
    title: str = "Accuracy ranking (blind judge)",
    height: int = 240,
) -> go.Figure:
    """Horizontal accuracy bars for Automated Benchmark (judge weighted %)."""
    if not ranking_rows:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title, height=height))
        return fig

    # Highest accuracy at top
    rows = sorted(ranking_rows, key=lambda r: float(r.get("accuracy") or 0), reverse=True)
    labels = []
    for r in rows:
        key = r.get("key") or ""
        short = {"chatgpt": "ChatGPT", "claude": "Claude", "gemini": "Gemini", "qvac": "QVAC"}.get(
            key, r.get("label") or key
        )
        labels.append(short)
    accs = [float(r.get("accuracy") or 0) for r in rows]
    colors = [MODEL_COLORS.get(r.get("key"), "#94a3b8") for r in rows]
    ranks = [int(r.get("rank") or i + 1) for i, r in enumerate(rows)]

    fig = go.Figure(
        go.Bar(
            y=labels[::-1],
            x=accs[::-1],
            orientation="h",
            marker=dict(color=colors[::-1], line=dict(color="#1e293b", width=1)),
            text=[f"#{r} · {a:.1f}%" for r, a in zip(ranks[::-1], accs[::-1])],
            textposition="outside",
            cliponaxis=False,
        )
    )
    layout = _base_layout(title, height=height)
    layout["margin"] = dict(l=56, r=72, t=48, b=28)
    layout["xaxis"] = dict(range=[0, 110], title="Accuracy %")
    layout["yaxis"] = dict(title="")
    fig.update_layout(**layout)
    return fig


def fig_judge_mean_accuracy_bars(
    ranking_mean_rows: list,
    *,
    title: str = "Multi-run mean accuracy ± std",
    height: int = 280,
) -> go.Figure:
    """Mean accuracy with ±1 std error bars (multi-run summary)."""
    if not ranking_mean_rows:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title, height=height))
        return fig

    rows = sorted(
        ranking_mean_rows,
        key=lambda r: float(r.get("accuracy_mean") or 0),
        reverse=True,
    )
    short = {
        "chatgpt": "ChatGPT",
        "claude": "Claude",
        "gemini": "Gemini",
        "qvac": "QVAC",
    }
    labels = [short.get(r.get("key"), r.get("key") or "?") for r in rows]
    means = [float(r.get("accuracy_mean") or 0) for r in rows]
    stds = [float(r.get("std") or 0) for r in rows]
    cvs = [float(r.get("cv_pct") or 0) for r in rows]
    colors = [MODEL_COLORS.get(r.get("key"), "#94a3b8") for r in rows]
    ranks = [int(r.get("rank") or i + 1) for i, r in enumerate(rows)]

    # Keep bar labels short so ±std whiskers do not cover the numbers.
    # Full Mean / ±Std / CV / Reliability live in the table under the chart.
    fig = go.Figure(
        go.Bar(
            y=labels[::-1],
            x=means[::-1],
            orientation="h",
            marker=dict(color=colors[::-1], line=dict(color="#1e293b", width=1)),
            error_x=dict(
                type="data",
                array=stds[::-1],
                visible=True,
                color="rgba(148,163,184,0.85)",
                thickness=1.2,
                width=5,
            ),
            text=[f"{m:.1f}%" for m in means[::-1]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#0f172a", size=13, family="Inter, system-ui, sans-serif"),
            hovertemplate=(
                "%{y}<br>Mean %{x:.1f}% ± %{customdata[0]:.1f}"
                "<br>CV %{customdata[1]:.0f}%<extra></extra>"
            ),
            customdata=list(zip(stds[::-1], cvs[::-1])),
            cliponaxis=False,
        )
    )
    layout = _base_layout(title, height=height)
    layout["margin"] = dict(l=72, r=28, t=48, b=36)
    xmax = max(means[i] + stds[i] for i in range(len(means))) if means else 100
    layout["xaxis"] = dict(
        range=[0, min(110, max(40, xmax * 1.12))],
        title="Mean accuracy %  (±1 std whiskers)",
        gridcolor="rgba(51,65,85,0.5)",
    )
    layout["yaxis"] = dict(title="")
    fig.update_layout(**layout)
    return fig
