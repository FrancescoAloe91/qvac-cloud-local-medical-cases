"""Shared visual layer: global CSS theme + small HTML rendering helpers.

Keeping this in one place lets app.py stay focused on flow/logic while the
look & feel (glassmorphism, gradients, animations, score badges, KPI tiles)
is defined and reused consistently everywhere.
"""

from __future__ import annotations

import html
import re
from typing import Optional

import pandas as pd
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

:root {
    --bg-0: #070a12;
    --bg-1: #0b0f1a;
    --panel: rgba(22, 27, 41, 0.72);
    --panel-solid: #131826;
    --border: rgba(148, 163, 184, 0.16);
    --border-strong: rgba(148, 163, 184, 0.3);
    --accent: #00d09c;
    --accent-2: #3b82f6;
    --accent-3: #a855f7;
    --text: #eef2f8;
    --text-dim: #b4c2d4;
    --text-muted: #8fa3bb;
    --radius: 16px;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', 'Inter', sans-serif !important; letter-spacing: -0.01em; }

/* ---------- animated backdrop ---------- */
.stApp {
    background: radial-gradient(circle at 15% 0%, rgba(0,208,156,0.10), transparent 45%),
                radial-gradient(circle at 85% 8%, rgba(59,130,246,0.10), transparent 45%),
                radial-gradient(circle at 50% 100%, rgba(168,85,247,0.07), transparent 55%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, #0a0d16 100%);
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.035) 50%, transparent 51%),
        radial-gradient(1px 1px at 70% 65%, rgba(255,255,255,0.03) 50%, transparent 51%);
    background-size: 340px 340px, 260px 260px;
    opacity: 0.6;
}

header[data-testid="stHeader"],
.stApp > header {
    display: none !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}
.block-container { padding-top: 1.4rem; padding-bottom: 1.4rem; max-width: 1480px; }
[data-testid="stMainBlockContainer"] { padding-top: 0.5rem; }
[data-testid="stAppViewContainer"] > section.main > div {
    padding-top: 0.75rem;
}

/* Global readable captions & secondary text */
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] {
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
    line-height: 1.45 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    line-height: 1.45;
}
.stInfo, .stSuccess, .stWarning {
    font-size: 0.84rem !important;
    line-height: 1.45 !important;
}

/* tighten Streamlit's own flex-gap between stacked elements everywhere */
[data-testid="stVerticalBlock"] { gap: 0.45rem !important; }
[data-testid="stHorizontalBlock"] {
    gap: 0.85rem !important;
    align-items: stretch;
    flex-wrap: wrap !important;
}
[data-testid="column"] {
    min-width: 0 !important;
    flex: 1 1 260px !important;
    overflow-wrap: anywhere;
    word-break: break-word;
}
[data-testid="column"] > div {
    min-width: 0 !important;
}
div.element-container { margin-bottom: 0 !important; }
[data-testid="stExpander"] details summary { padding: 0.28rem 0.65rem !important; }
[data-testid="stExpander"] .streamlit-expanderContent { padding-top: 0.25rem !important; }
[data-testid="stMetric"] { margin-bottom: 0 !important; }
[data-testid="stMetricValue"] { font-size: 1.05rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.28rem !important; }
[data-testid="stSidebar"] hr { margin: 0.45rem 0 !important; }
[data-testid="stSidebar"] div.element-container { margin-bottom: 0 !important; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 0.3rem !important;
}
[data-testid="stSidebar"] [data-testid="column"] {
    min-width: 0 !important;
    align-self: center !important;
}
[data-testid="stSidebar"] .stButton { margin: 0 !important; padding: 0 !important; width: 100%; }
[data-testid="stSidebar"] .stButton button {
    padding: 0.2rem 0.35rem !important;
    min-height: 1.55rem !important;
    max-height: 1.65rem !important;
    font-size: 0.68rem !important;
    line-height: 1.1 !important;
    white-space: nowrap !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { margin: 0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin: 0 !important; line-height: 1.2; }
[data-testid="stSidebar"] {
    min-width: 300px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.55rem !important;
    padding-bottom: 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0.28rem 0.42rem !important;
    margin-bottom: 0.22rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.slot-state-) {
    padding: 0.22rem 0.38rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.slot-state-) > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.slot-state-) [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 0.2rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.usdt-wallet-marker) {
    border-color: rgba(245, 197, 24, 0.62) !important;
    background: linear-gradient(135deg, rgba(245,197,24,0.18), rgba(251,191,36,0.06)) !important;
    margin-bottom: 0.35rem !important;
    padding: 0.35rem 0.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.slot-state-filled) {
    border-color: rgba(0, 208, 156, 0.45) !important;
    background: rgba(0, 208, 156, 0.06) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.slot-state-viewing) {
    border-color: var(--accent) !important;
    background: rgba(0, 208, 156, 0.1) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.slot-state-empty) {
    border-style: dashed !important;
    border-color: rgba(148, 163, 184, 0.35) !important;
    background: rgba(15, 20, 32, 0.35) !important;
}
.sidebar-brand {
    font-size: 0.88rem;
    font-weight: 800;
    line-height: 1.25;
    margin-bottom: 0.4rem;
    color: var(--text);
}
.sidebar-brand b { color: var(--text); }
.sidebar-section-title {
    font-size: 0.8rem;
    font-weight: 700;
    margin: 0.1rem 0 0.3rem;
    color: var(--text);
    line-height: 1.25;
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] { padding: 0.45rem 0.65rem !important; }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes glowPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(0,208,156,0.35); } 50% { box-shadow: 0 0 0 7px rgba(0,208,156,0); } }
@keyframes shimmer { 0% { background-position: -300px 0; } 100% { background-position: 300px 0; } }
@keyframes floatY { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes stepGlow { 0%,100% { box-shadow: 0 0 0 0 rgba(0,208,156,0.45); } 50% { box-shadow: 0 0 0 6px rgba(0,208,156,0); } }
@keyframes lineGrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes rewardPop { 0% { transform: scale(0.55); opacity: 0; } 55% { transform: scale(1.12); opacity: 1; } 100% { transform: scale(1); opacity: 1; } }
@keyframes coinSpin { 0% { transform: rotateY(0deg) scale(0.4); opacity: 0; } 60% { opacity: 1; } 100% { transform: rotateY(360deg) scale(1); opacity: 1; } }
@keyframes rewardRing { 0% { box-shadow: 0 0 0 0 rgba(0,208,156,0.45); } 100% { box-shadow: 0 0 0 18px rgba(0,208,156,0); } }

.fade-in { animation: fadeInUp 0.45s cubic-bezier(.22,1,.36,1) both; }
.fade-in.d1 { animation-delay: .04s; } .fade-in.d2 { animation-delay: .08s; }
.fade-in.d3 { animation-delay: .12s; } .fade-in.d4 { animation-delay: .16s; }
.fade-in.d5 { animation-delay: .20s; }

/* ---------- headline (compact) ---------- */
.app-title {
    font-size: 1.55rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.2;
    background: linear-gradient(100deg, #00d09c 0%, #3b82f6 45%, #a855f7 90%);
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: gradientShift 10s ease infinite;
    display: flex; align-items: center; gap: 8px;
}
.app-subtitle { color: var(--text-dim); font-size: 0.82rem; margin: 0.15rem 0 0.5rem; line-height: 1.4; }
.live-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(0,208,156,0.12); border: 1px solid rgba(0,208,156,0.35);
    color: #6ee7c8; font-size: 0.66rem; font-weight: 700; padding: 3px 10px;
    border-radius: 999px; letter-spacing: 0.02em;
    transition: transform 0.25s cubic-bezier(.22,1,.36,1);
    max-width: 100%; white-space: normal; line-height: 1.35;
}
.live-chip:hover { transform: translateY(-1px); }
.live-dot { width: 5px; height: 5px; border-radius: 50%; background: #00d09c; animation: glowPulse 1.8s infinite; }

/* ---------- compact eyebrow (replaces bulky numbered section headers) ---------- */
.eyebrow {
    display: flex; align-items: center; gap: 6px; margin: 0.45rem 0 0.28rem;
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.07em; text-transform: uppercase;
    color: #c5d0df;
}
.eyebrow .eyebrow-icon { font-size: 0.78rem; opacity: 0.9; }
.eyebrow::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border-strong), transparent); }

/* ---------- progress stepper ---------- */
.stepper { display: flex; align-items: center; margin: 0.05rem 0 0.45rem; }
.stepper .step { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
.stepper .node {
    width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.62rem; font-weight: 800; color: var(--text-dim);
    background: rgba(148,163,184,0.12); border: 1px solid var(--border-strong);
    transition: all 0.35s cubic-bezier(.22,1,.36,1);
}
.stepper .step.done .node { background: var(--accent); border-color: var(--accent); color: #04231a; }
.stepper .step.current .node { background: linear-gradient(135deg, var(--accent), var(--accent-2)); border-color: transparent; color: #04231a; animation: stepGlow 2s infinite; }
.stepper .label { font-size: 0.68rem; font-weight: 700; color: var(--text-dim); white-space: normal; line-height: 1.2; max-width: 10rem; }
.stepper .step.done .label, .stepper .step.current .label { color: var(--text); }
.stepper .connector { flex: 1; height: 2px; background: var(--border); margin: 0 8px; border-radius: 2px; position: relative; overflow: hidden; min-width: 16px; }
.stepper .connector.filled::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, var(--accent), var(--accent-2)); transform-origin: left; animation: lineGrow 0.5s cubic-bezier(.22,1,.36,1) both; }

/* ---------- generic glass panel ---------- */
.glass-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.6rem 0.85rem;
    backdrop-filter: blur(6px);
    transition: border-color 0.3s cubic-bezier(.22,1,.36,1), transform 0.3s cubic-bezier(.22,1,.36,1), box-shadow 0.3s cubic-bezier(.22,1,.36,1);
}
.glass-panel:hover { border-color: var(--border-strong); }

/* ---------- compact case picker ---------- */
.case-info-bar {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    background: linear-gradient(150deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01));
    border: 1px solid var(--border);
    border-left: 3px solid var(--case-color, var(--accent));
    border-radius: 12px;
    padding: 0.5rem 0.8rem;
    transition: border-color 0.3s cubic-bezier(.22,1,.36,1);
}
.case-info-icon {
    font-size: 1rem; line-height: 1; flex-shrink: 0;
    width: 26px; height: 26px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    background: color-mix(in srgb, var(--case-color, var(--accent)) 18%, transparent);
}
.case-specialty {
    display: inline-block; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
    color: var(--case-color, var(--accent)); background: color-mix(in srgb, var(--case-color, var(--accent)) 16%, transparent);
    padding: 2px 7px; border-radius: 999px;
}
.case-info-focus { font-size: 0.76rem; color: #cbd5e1; flex: 1 1 180px; min-width: 0; line-height: 1.35; }
.case-info-title { font-weight: 700; font-size: 0.82rem; color: var(--text); }

/* ---------- model output cards (compact) ---------- */
.model-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 3px solid var(--model-color, var(--accent));
    border-radius: 12px;
    padding: 0.65rem 0.8rem 0.5rem;
    margin-bottom: 0.4rem;
    backdrop-filter: blur(6px);
    transition: border-color 0.3s cubic-bezier(.22,1,.36,1);
}
.model-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.model-card-name { font-weight: 800; font-size: 0.88rem; color: var(--text); display: flex; align-items: center; gap: 5px; flex-wrap: wrap; min-width: 0; flex: 1 1 140px; }
.model-card-name .m-icon {
    font-size: 0.8rem; width: 22px; height: 22px; border-radius: 7px;
    display: inline-flex; align-items: center; justify-content: center;
    background: color-mix(in srgb, var(--model-color, var(--accent)) 16%, transparent);
    flex-shrink: 0;
}
.card-link-pill {
    font-size: 0.62rem; font-weight: 700; padding: 1px 8px; border-radius: 999px;
    background: rgba(148,163,184,0.12); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.3);
    text-decoration: none; flex-shrink: 0; transition: background 0.2s, border-color 0.2s, color 0.2s;
}
.card-link-pill:hover { background: color-mix(in srgb, var(--model-color, var(--accent)) 22%, transparent); color: var(--text); border-color: var(--model-color, var(--accent)); }
.model-vendor { color: var(--text-dim); font-size: 0.72rem; font-weight: 500; line-height: 1.35; word-wrap: break-word; overflow-wrap: anywhere; }
.model-vendor .cloud-tier-tag { color: #cbd5e1; }
.model-instructions { font-size: 0.78rem; color: #d8e2ef; margin: 0.35rem 0 0.25rem; line-height: 1.45; }
.sidebar-footer-note {
    color: var(--text-dim); font-size: 0.68rem; line-height: 1.45; margin: 0.55rem 0 0.15rem;
    padding: 0.35rem 0.1rem 0.1rem; word-wrap: break-word; overflow-wrap: anywhere;
    border-top: 1px solid rgba(148,163,184,0.14);
}
[data-testid="stSidebar"] .sidebar-footer-note { max-width: 100%; }

.status-pill { display: inline-flex; align-items: center; gap: 4px; font-size: 0.68rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; flex-shrink: 0; line-height: 1.2; }
.status-empty { background: rgba(148,163,184,0.18); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }
.status-filled { background: rgba(0,208,156,0.18); color: #86efcf; border: 1px solid rgba(0,208,156,0.42); }

/* ---------- ranking narrative cards (compact) ---------- */
.narrative-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--model-color, var(--accent));
    border-radius: 10px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.28rem;
    transition: border-color 0.25s cubic-bezier(.22,1,.36,1);
}
.narrative-card:hover { border-color: var(--border-strong); }
.narrative-card.tone-strength { background: linear-gradient(90deg, rgba(0,208,156,0.06), transparent 55%), var(--panel); }
.narrative-card.tone-watch { background: linear-gradient(90deg, rgba(245,158,11,0.05), transparent 55%), var(--panel); }
.narrative-head { display: flex; align-items: center; gap: 6px; margin-bottom: 0.12rem; flex-wrap: nowrap; }
.narrative-rank { font-size: 0.9rem; line-height: 1; flex-shrink: 0; }
.narrative-name { font-weight: 700; font-size: 0.78rem; color: var(--text); display: flex; align-items: center; gap: 4px; flex: 1 1 auto; min-width: 0; white-space: nowrap; }
.narrative-head .score-badge { flex-shrink: 0; font-size: 0.68rem !important; padding: 2px 6px !important; }
.narrative-name .m-icon {
    font-size: 0.68rem; width: 18px; height: 18px; border-radius: 5px;
    display: inline-flex; align-items: center; justify-content: center;
    background: color-mix(in srgb, var(--model-color, var(--accent)) 16%, transparent);
    flex-shrink: 0;
}
.narrative-blurb { margin: 0; font-size: 0.7rem; color: #b4c2d4; line-height: 1.35; }
.narrative-blurb strong { color: var(--text); font-weight: 600; }
.narrative-blurb + .narrative-blurb { margin-top: 0.1rem; }

/* ---------- privacy strip (readable scale 0–100) ---------- */
.privacy-strip {
    margin: 0.35rem 0 0.5rem;
    padding: 0.5rem 0.65rem 0.55rem;
    background: linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid var(--border);
    border-radius: 12px;
}
.privacy-strip-head {
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.35rem;
    letter-spacing: 0.03em;
    line-height: 1.25;
}
.privacy-strip-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.45rem;
}
.privacy-chip {
    min-width: 0;
    padding: 0.35rem 0.42rem 0.38rem;
    border-radius: 9px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(148,163,184,0.18);
}
.privacy-chip--zero {
    border-color: rgba(239, 68, 68, 0.35);
    background: rgba(239, 68, 68, 0.06);
}
.privacy-chip--full {
    border-color: rgba(0, 208, 156, 0.45);
    background: rgba(0, 208, 156, 0.08);
}
.privacy-chip-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    line-height: 1.15;
    margin-bottom: 0.28rem;
}
.privacy-chip-name {
    font-size: 0.74rem;
    font-weight: 800;
    color: var(--pc, var(--text));
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.privacy-chip-val {
    font-size: 0.92rem;
    font-weight: 900;
    color: var(--text);
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
}
.privacy-chip--zero .privacy-chip-val { color: #fca5a5; }
.privacy-chip--full .privacy-chip-val { color: #6ee7b7; }
.privacy-chip-track {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 0.28rem;
}
.privacy-scale-edge {
    font-size: 0.62rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    min-width: 1.1rem;
}
.privacy-scale-edge--min {
    color: #f87171;
    text-align: right;
}
.privacy-scale-edge--max {
    color: #34d399;
    text-align: left;
}
.privacy-chip-bar {
    height: 9px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(239,68,68,0.22) 0%, rgba(148,163,184,0.16) 50%, rgba(0,208,156,0.28) 100%);
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.12);
}
.privacy-chip-bar span {
    display: block;
    height: 100%;
    background: var(--pc, var(--accent));
    border-radius: 999px;
    min-width: 2px;
    box-shadow: 0 0 8px color-mix(in srgb, var(--pc, var(--accent)) 45%, transparent);
}
.privacy-strip-foot {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    margin-top: 0.32rem;
    font-size: 0.64rem;
    font-weight: 700;
    color: var(--text-dim);
    line-height: 1.3;
}
.privacy-strip-foot span:first-child { color: #fca5a5; }
.privacy-strip-foot span:last-child { color: #6ee7b7; text-align: right; }
.privacy-strip-caption {
    margin: 0.28rem 0 0;
    font-size: 0.68rem;
    color: #94a3b8;
    line-height: 1.35;
}

/* ---------- score story (demo / KPI weights) ---------- */
.score-story {
    margin: 0.35rem 0 0.45rem;
    padding: 0.55rem 0.65rem;
    border-radius: 12px;
    border: 1px solid rgba(96, 165, 250, 0.35);
    background: linear-gradient(145deg, rgba(59, 130, 246, 0.12), rgba(15, 23, 42, 0.35));
}
.score-story-title {
    font-size: 0.76rem;
    font-weight: 800;
    color: #bfdbfe;
    margin-bottom: 0.35rem;
    letter-spacing: 0.02em;
}
.score-story-block {
    margin: 0 0 0.32rem;
    font-size: 0.72rem;
    color: #cbd5e1;
    line-height: 1.45;
}
.score-story-block:last-child { margin-bottom: 0; }
.score-story-block strong { color: #f1f5f9; font-weight: 700; }

/* ---------- tier / model badges ---------- */
.tier-light  { background:rgba(107,114,128,0.22); color:#cbd5e1; padding:2px 8px; border:1px solid rgba(148,163,184,0.35); border-radius:999px; font-size:0.62rem; font-weight:700; }
.tier-medium { background:rgba(59,130,246,0.18); color:#93c5fd; padding:2px 8px; border:1px solid rgba(59,130,246,0.4); border-radius:999px; font-size:0.62rem; font-weight:700; }
.tier-premium{ background:rgba(245,197,24,0.16); color:#fde68a; padding:2px 8px; border:1px solid rgba(245,197,24,0.4); border-radius:999px; font-size:0.62rem; font-weight:700; }
.tier-local  { background:rgba(0,208,156,0.16); color:#5eead4; padding:2px 8px; border:1px solid rgba(0,208,156,0.4); border-radius:999px; font-size:0.62rem; font-weight:700; }

/* ---------- KPI command-center tiles (compact) ---------- */
.kpi-tile {
    background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.4rem 0.6rem;
    height: 100%;
    transition: transform 0.3s cubic-bezier(.22,1,.36,1), border-color 0.3s cubic-bezier(.22,1,.36,1), box-shadow 0.3s cubic-bezier(.22,1,.36,1);
}
.kpi-tile:hover { transform: translateY(-3px); border-color: var(--border-strong); box-shadow: 0 10px 22px -12px rgba(0,0,0,0.5); }
.kpi-label { font-size: 0.64rem; color: var(--text-dim); font-weight: 700; margin-bottom: 0.15rem; text-transform: uppercase; letter-spacing: 0.02em; line-height: 1.25; white-space: normal; }
.kpi-value { font-size: 1.28rem; font-weight: 800; color: var(--text); line-height: 1.2; word-break: break-word; }
.kpi-sub { font-size: 0.64rem; color: var(--text-dim); margin-top: 0.15rem; }
.kpi-value.accent { color: var(--accent); }

/* ---------- score badge + bar ---------- */
.score-badge {
    display: inline-flex; align-items: center; gap: 6px; font-weight: 800; font-size: 0.85rem;
    padding: 2px 4px;
}
.score-bar-wrap { width: 100%; height: 6px; border-radius: 999px; background: rgba(148,163,184,0.14); overflow: hidden; margin-top: 4px; }
.score-bar-fill { height: 100%; border-radius: 999px; background-size: 300px 100%; animation: shimmer 2.4s linear infinite; transition: width 0.5s cubic-bezier(.22,1,.36,1); }

/* ---------- misc widget polish ---------- */
[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.35rem 0.6rem; backdrop-filter: blur(4px);
    transition: transform 0.3s cubic-bezier(.22,1,.36,1);
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }
[data-testid="stMetric"] label {
    font-size: 0.66rem !important;
    color: var(--text-dim) !important;
    white-space: normal !important;
    line-height: 1.25 !important;
}
[data-testid="stMetricValue"] { font-size: 1.05rem !important; }

div[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 10px; background: rgba(255,255,255,0.015); }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; font-size: 0.85rem; padding: 0.35rem 0.7rem; transition: color 0.25s ease; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }

/* segmented control (case picker / tier picker) — compact pills */
div[data-testid="stSegmentedControl"] button {
    border-radius: 999px !important; font-size: 0.78rem !important;
    transition: transform 0.22s cubic-bezier(.22,1,.36,1), background 0.22s ease, box-shadow 0.22s ease !important;
}
div[data-testid="stSegmentedControl"] button:hover { transform: translateY(-1px); }
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    box-shadow: 0 4px 14px -6px rgba(0,208,156,0.55);
    animation: fadeInUp 0.3s cubic-bezier(.22,1,.36,1) both;
}

.stButton > button, .stDownloadButton > button, .stLinkButton > a, .stPopover > button {
    border-radius: 9px !important; transition: transform 0.22s cubic-bezier(.22,1,.36,1), box-shadow 0.22s cubic-bezier(.22,1,.36,1), border-color 0.22s ease !important;
    font-size: 0.84rem !important;
}
.stButton > button:hover, .stLinkButton > a:hover, .stPopover > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px -8px rgba(0,0,0,0.5); }
.stButton > button[kind="primary"] { animation: glowPulse 2.8s infinite; font-weight: 700 !important; }

textarea, .stTextArea textarea {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    line-height: 1.45 !important;
}

hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.25); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.4); }

.footer-note { color: var(--text-dim); font-size: 0.68rem; text-align: center; margin-top: 1.2rem; opacity: 0.8; }

/* ---------- closing "decision" section ---------- */
.decision-lead { color: #cbd5e1; font-size: 0.86rem; margin: 0 0 0.6rem; line-height: 1.45; }
.decision-caption { color: var(--text-dim); font-size: 0.68rem; margin: 0.35rem 0 0; line-height: 1.4; }
.decision-col { min-width: 0; }
.decision-col--usdt {
    background: rgba(245, 197, 24, 0.1);
    border: 1px solid rgba(245, 197, 24, 0.42);
    border-radius: 10px;
    padding: 0.45rem 0.5rem 0.55rem;
}

/* ---------- USDT / wallet zone (yellow highlight) ---------- */
.usdt-zone.wallet-panel {
    border: 1.5px solid rgba(245, 197, 24, 0.58);
    background: linear-gradient(160deg, rgba(245,197,24,0.18), rgba(251,191,36,0.06));
    border-radius: 12px;
    padding: 0.55rem 0.65rem 0.6rem;
    margin: 0.15rem 0 0.35rem;
    box-shadow: 0 4px 16px -10px rgba(245, 197, 24, 0.45);
}
.usdt-zone.wallet-compact {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0;
    margin: 0;
    border: none;
    background: transparent;
    box-shadow: none;
    white-space: nowrap;
}
.wallet-compact-left {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    min-width: 0;
}
.wallet-compact-icon { font-size: 0.9rem; line-height: 1; flex-shrink: 0; }
.wallet-compact-label {
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #fde68a;
}
.wallet-compact-bal {
    font-size: 0.88rem;
    font-weight: 800;
    color: #fef3c7;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
}
.wallet-compact-bal span {
    font-size: 0.68rem;
    color: #fbbf24;
    font-weight: 700;
}
.usdt-zone-head {
    display: flex; align-items: center; gap: 6px; margin-bottom: 0.25rem;
}
.usdt-zone-icon { font-size: 1rem; line-height: 1; }
.usdt-zone-title {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em;
    text-transform: uppercase; color: #fde68a;
}
.usdt-balance {
    font-size: 1.45rem; font-weight: 800; color: #fef3c7; line-height: 1.1;
}
.usdt-currency { font-size: 0.82rem; color: #fbbf24; font-weight: 700; }
.usdt-zone-caption {
    margin-top: 0.25rem; font-size: 0.66rem; color: rgba(253, 230, 138, 0.85); line-height: 1.35;
}
.usdt-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(245, 197, 24, 0.16); border: 1px solid rgba(245, 197, 24, 0.48);
    color: #fde68a; font-size: 0.66rem; font-weight: 700; padding: 2px 9px;
    border-radius: 999px; letter-spacing: 0.02em;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.usdt-decision-marker) {
    border-color: rgba(245, 197, 24, 0.58) !important;
    background: linear-gradient(160deg, rgba(245,197,24,0.14), rgba(251,191,36,0.04)) !important;
    border-width: 1.5px !important;
    border-radius: 14px !important;
    padding: 0.55rem 0.7rem 0.65rem !important;
    margin: 0.35rem 0 0.85rem !important;
    box-shadow: 0 4px 18px -10px rgba(245, 197, 24, 0.38) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.usdt-decision-marker) [data-testid="column"]:nth-child(3) {
    background: rgba(245, 197, 24, 0.12);
    border: 1px solid rgba(245, 197, 24, 0.42);
    border-radius: 10px;
    padding: 0.35rem 0.45rem 0.5rem;
}
[data-testid="stSidebar"] .usdt-zone.wallet-panel {
    margin-top: 0.25rem;
}

@media (max-width: 1100px) {
    .stepper { flex-wrap: wrap; row-gap: 0.35rem; }
    .stepper .connector { flex: 0 0 12px; min-width: 12px; }
    .app-title { font-size: 1.35rem; }
}

/* ---------- reward / payment success animation ---------- */
.reward-pop {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
    padding: 0.75rem 1rem 0.9rem; text-align: center;
    animation: rewardPop 0.5s cubic-bezier(.22,1.6,.36,1) both;
}
.reward-coin {
    font-size: 2.1rem; width: 58px; height: 58px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,208,156,0.14); border: 1px solid rgba(0,208,156,0.4);
    animation: coinSpin 0.7s ease-out both, rewardRing 1.4s ease-out 0.5s;
    margin-bottom: 4px;
}
.reward-amount { font-size: 2rem; font-weight: 800; color: var(--accent); line-height: 1.1; }
.reward-balance { font-size: 0.76rem; color: var(--text-dim); margin-top: 2px; }
.reward-balance b { color: var(--text); font-weight: 700; }

/* ---------- live local-inference stream (real tokens, not a fake typewriter) ---------- */
.qvac-live-stream {
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 0.78rem; line-height: 1.5; color: #b7f7e4;
    background: rgba(0,208,156,0.06); border: 1px solid rgba(0,208,156,0.28);
    border-radius: 10px; padding: 0.65rem 0.8rem; max-height: 220px;
    overflow-y: auto; white-space: pre-wrap; word-break: break-word;
}
.qvac-live-stream::-webkit-scrollbar { width: 6px; }
.qvac-live-meta {
    font-size: 0.7rem; color: var(--text-dim); margin-top: 0.3rem;
    display: flex; align-items: center; gap: 6px;
}
.qvac-live-metrics-bar {
    font-size: 0.82rem; color: #e2e8f0;
    background: linear-gradient(135deg, rgba(0,208,156,0.12), rgba(56,189,248,0.08));
    border: 1px solid rgba(0,208,156,0.35);
    border-radius: 10px; padding: 0.55rem 0.85rem;
    margin: 0.4rem 0 0.6rem 0;
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.65rem;
    letter-spacing: 0.01em;
}
.qvac-live-metrics-bar b { color: #6ee7b7; font-weight: 700; }
.qvac-live-metrics-done { border-color: rgba(34,197,94,0.45); }

/* ---------- sidebar case slots ---------- */
.slot-header {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    width: 100%;
    min-height: 1.55rem;
    margin: 0;
}
.slot-n {
    font-weight: 800;
    font-size: 0.72rem;
    color: var(--text-dim);
    flex-shrink: 0;
    width: 0.85rem;
}
.slot-name {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--text);
}
.slot-meta {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    flex-shrink: 0;
}
.slot-time {
    font-size: 0.66rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #f1f5f9;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(203, 213, 225, 0.3);
    padding: 0 5px;
    border-radius: 3px;
    line-height: 1.35;
}
.slot-runs {
    font-size: 0.62rem;
    font-weight: 700;
    color: #cbd5e1;
}
.slot-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.slot-dot--empty { background: #64748b; }
.slot-dot--filled { background: #34d399; }
.slot-dot--viewing { background: #60a5fa; box-shadow: 0 0 0 2px rgba(96,165,250,0.35); }

/* ---------- sidebar finale ranking (highlight) ---------- */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-finale-marker) {
    margin-top: 0.35rem !important;
    margin-bottom: 0.35rem !important;
    border-color: rgba(245, 158, 11, 0.55) !important;
    background: linear-gradient(145deg, rgba(245, 158, 11, 0.24), rgba(234, 88, 12, 0.14)) !important;
    box-shadow: inset 0 1px 0 rgba(253, 224, 71, 0.22);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-finale-marker) button {
    color: #1c1917 !important;
    border-color: rgba(253, 224, 71, 0.75) !important;
    background: linear-gradient(135deg, #fbbf24 0%, #f97316 100%) !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-finale-marker) button:hover {
    color: #0c0a09 !important;
    background: linear-gradient(135deg, #fde047 0%, #fb923c 100%) !important;
    border-color: #fef08a !important;
    box-shadow: 0 0 0 1px rgba(253, 224, 71, 0.35), 0 6px 18px -8px rgba(249, 115, 22, 0.65);
}

/* ---------- sidebar danger zone (clear saved only) ---------- */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-danger-zone-marker) {
    margin-top: 2rem !important;
    border-color: rgba(239, 68, 68, 0.45) !important;
    background: rgba(15, 10, 10, 0.35) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.sidebar-danger-zone-marker) [data-testid="stCaption"] p {
    color: #fca5a5 !important;
    font-size: 0.66rem !important;
    line-height: 1.35 !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-danger-zone-marker) button {
    color: #ef4444 !important;
    border-color: rgba(239, 68, 68, 0.7) !important;
    background: rgba(127, 29, 29, 0.35) !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-danger-zone-marker) button:hover {
    color: #fff !important;
    background: #b91c1c !important;
    border-color: #fca5a5 !important;
}

/* ---------- tables: readable text, compact rows ---------- */
[data-testid="stDataFrame"] div[data-testid="glideDataEditor"] {
    font-size: 0.82rem !important;
}
[data-testid="stDataFrame"] [role="gridcell"] {
    white-space: normal !important;
    line-height: 1.35 !important;
    word-break: break-word !important;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    white-space: normal !important;
    line-height: 1.25 !important;
    word-break: break-word !important;
    font-weight: 700 !important;
}

/* dialog: red confirm for destructive clear-saved */
[data-testid="stDialog"] [data-testid="stVerticalBlock"]:has(.dialog-danger-confirm-marker) button {
    background: #b91c1c !important;
    border: 1px solid #fca5a5 !important;
    color: #fff !important;
    font-weight: 800 !important;
}
[data-testid="stDialog"] [data-testid="stVerticalBlock"]:has(.dialog-danger-confirm-marker) button:hover {
    background: #dc2626 !important;
    border-color: #fff !important;
}

.narrative-name { flex-wrap: wrap; word-break: break-word; }
.model-card-name { flex-wrap: wrap; word-break: break-word; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_section_title_html(title: str, count: str) -> str:
    return f'<p class="sidebar-section-title">{html.escape(title)} · {html.escape(count)}</p>'


def case_slot_header_html(
    index: int,
    short: str,
    state: str,
    saved_at: str,
    lang: str,
    runs_tag: str = "",
) -> str:
    """Riga intestazione slot — una linea, niente sovrapposizioni."""
    from lib.session_store import format_saved_at

    dot_cls = {"empty": "empty", "filled": "filled", "viewing": "viewing"}.get(state, "empty")
    meta_parts = []
    if saved_at and state != "empty":
        time_str = format_saved_at(saved_at, lang)
        meta_parts.append(f'<span class="slot-time">{html.escape(time_str)}</span>')
    if runs_tag:
        meta_parts.append(f'<span class="slot-runs">{html.escape(runs_tag)}</span>')
    meta_parts.append(f'<span class="slot-dot slot-dot--{dot_cls}"></span>')
    meta_html = "".join(meta_parts)
    return (
        f'<span class="slot-state-{html.escape(state)}" style="display:none"></span>'
        f'<div class="slot-header">'
        f'<span class="slot-n">{index}</span>'
        f'<span class="slot-name">{html.escape(short)}</span>'
        f'<span class="slot-meta">{meta_html}</span>'
        f"</div>"
    )


def case_slot_tile_html(
    index: int,
    short: str,
    state: str,
    saved_at: str,
    lang: str,
    runs_tag: str = "",
) -> str:
    """Alias compatto."""
    return case_slot_header_html(index, short, state, saved_at, lang, runs_tag=runs_tag)


def case_slot_card_html(
    index: int,
    short: str,
    state: str,
    saved_at: str,
    lang: str,
    extra_line: str = "",
) -> str:
    """Legacy alias — usa il tile compatto."""
    return case_slot_tile_html(index, short, state, saved_at, lang, runs_tag=extra_line)


def score_color(value: float) -> str:
    if value is None:
        return "#64748b"
    if value >= 75:
        return "#00d09c"
    if value >= 45:
        return "#f5c518"
    return "#ef4444"


def score_badge_html(value: float, suffix: str = "%") -> str:
    color = score_color(value)
    return (
        f'<span class="score-badge" style="color:{color};">'
        f'<span class="score-bar-wrap" style="width:46px; display:inline-block; vertical-align:middle;">'
        f'<span class="score-bar-fill" style="width:{min(max(value or 0, 0), 100):.0f}%; '
        f'background:linear-gradient(90deg,{color}55,{color});"></span></span>'
        f'{value:.0f}{suffix}</span>'
    )


def kpi_tile_html(label: str, value: str, sub: str = "", accent: bool = False, delay: int = 0) -> str:
    cls = "kpi-value accent" if accent else "kpi-value"
    sub_html = f'<div class="kpi-sub">{html.escape(sub)}</div>' if sub else ""
    delay_cls = f" d{delay}" if delay else ""
    return (
        f'<div class="kpi-tile fade-in{delay_cls}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="{cls}">{value}</div>'
        f"{sub_html}</div>"
    )


def _md_bold(text: str) -> str:
    """Escape text then convert simple **bold** markers to <strong> tags.

    Escaping first neutralizes any stray angle brackets/quotes in the input
    (defense in depth even though callers only pass our own template
    strings), and only then are the literal ** markers turned into markup.
    """
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def privacy_strip_html(ranking_df: pd.DataFrame, lang: str = "en") -> str:
    """Privacy row with 0–100 scale per model — bigger than micro-strip, smaller than Plotly gauges."""
    from lib.i18n import t
    from lib.metrics import TABLE_MODEL_SHORT, _L
    from lib.tiers import MODEL_CONFIG

    if ranking_df.empty:
        return ""

    L = _L(lang)
    chips = []
    for _, row in ranking_df.iterrows():
        key = row.get("key", "")
        cfg = MODEL_CONFIG.get(key, {})
        try:
            pct = int(float(row.get(L["privacy"], 0) or 0))
        except (TypeError, ValueError):
            pct = 0
        pct = max(0, min(100, pct))
        color = cfg.get("color", "#94a3b8")
        name = TABLE_MODEL_SHORT.get(key, str(row.get(L["model"], key)))
        chip_cls = "privacy-chip--full" if pct >= 100 else ("privacy-chip--zero" if pct <= 0 else "")
        chips.append(
            f'<div class="privacy-chip {chip_cls}" style="--pc:{html.escape(color)};">'
            f'<div class="privacy-chip-top">'
            f'<span class="privacy-chip-name">{html.escape(name)}</span>'
            f'<span class="privacy-chip-val">{pct}%</span>'
            f"</div>"
            f'<div class="privacy-chip-track">'
            f'<span class="privacy-scale-edge privacy-scale-edge--min">{html.escape(t("chart.privacy_scale_low", lang))}</span>'
            f'<div class="privacy-chip-bar"><span style="width:{pct}%;"></span></div>'
            f'<span class="privacy-scale-edge privacy-scale-edge--max">{html.escape(t("chart.privacy_scale_high", lang))}</span>'
            f"</div>"
            f"</div>"
        )

    return (
        f'<div class="privacy-strip fade-in">'
        f'<div class="privacy-strip-head">🔒 {html.escape(t("chart.privacy_title", lang))}</div>'
        f'<div class="privacy-strip-row">{"".join(chips)}</div>'
        f'<div class="privacy-strip-foot">'
        f'<span>0% · {html.escape(t("chart.privacy_scale_cloud", lang))}</span>'
        f'<span>100% · {html.escape(t("chart.privacy_scale_local", lang))}</span>'
        f"</div>"
        f'<p class="privacy-strip-caption">{_md_bold(t("chart.privacy_caption", lang))}</p>'
        f"</div>"
    )


def score_story_html(use_gold: bool, lang: str = "en") -> str:
    """Demo-friendly explanation of 0–100 scores and weighted KPIs."""
    from lib.i18n import t

    blocks = [f'<p class="score-story-block">{_md_bold(t("ranking.score_story_gold_intro" if use_gold else "ranking.score_story_consensus_intro", lang))}</p>']
    if use_gold:
        blocks.append(f'<p class="score-story-block">{_md_bold(t("ranking.score_story_gold_clinical", lang))}</p>')
        blocks.append(f'<p class="score-story-block">{_md_bold(t("ranking.score_story_gold_consensus", lang))}</p>')
    else:
        blocks.append(f'<p class="score-story-block">{_md_bold(t("ranking.score_story_consensus_formula", lang))}</p>')

    return (
        f'<div class="score-story fade-in">'
        f'<div class="score-story-title">📖 {html.escape(t("ranking.score_story_title", lang))}</div>'
        f'{"".join(blocks)}'
        f"</div>"
    )


def narrative_card_html(
    rank: int, icon: str, color: str, name: str, score: float, bullets: list, tone: str = "neutral"
) -> str:
    """Compact one-line (max two) takeaway per model."""
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    blurbs = "".join(f'<p class="narrative-blurb">{_md_bold(b)}</p>' for b in (bullets or [])[:2])
    return (
        f'<div class="narrative-card fade-in tone-{tone}" style="--model-color:{color};">'
        f'<div class="narrative-head">'
        f'<span class="narrative-rank">{medal}</span>'
        f'<span class="narrative-name"><span class="m-icon">{icon}</span>{html.escape(name)}</span>'
        f"{score_badge_html(score)}"
        f"</div>"
        f"{blurbs}"
        f"</div>"
    )


def table_height(n_rows: int, row_px: int = 48, header_px: int = 42, pad_px: int = 6) -> int:
    """Exact pixel height to show every row of a dataframe with no internal
    scrollbar — used for the small (2-5 model) KPI tables where an inner
    scroll arrow only hides data the user should see at a glance."""
    return header_px + row_px * max(n_rows, 1) + pad_px


def model_table_column_config(lang: str) -> dict:
    """Streamlit column widths for Model + Version (full tier text, no clipping)."""
    from lib.metrics import _L

    L = _L(lang)
    return {
        L["model"]: st.column_config.TextColumn(L["model"], width="small"),
        L["version"]: st.column_config.TextColumn(
            L["version"], width="medium", help=None
        ),
    }


def show_model_table(
    df: pd.DataFrame,
    lang: str,
    tier_labels: Optional[dict] = None,
    columns: Optional[list] = None,
    **dataframe_kwargs,
) -> None:
    """Render a KPI table with Model + Version columns (no forced height — avoids blank rows)."""
    from lib.metrics import prepare_model_table

    display = prepare_model_table(df, lang, tier_labels, columns=columns)
    if display.empty:
        return
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=model_table_column_config(lang),
        **dataframe_kwargs,
    )


def eyebrow_html(icon: str, title: str) -> str:
    """Compact single-line section label — replaces the old bulky numbered header."""
    return (
        '<div class="eyebrow fade-in">'
        f'<span class="eyebrow-icon">{icon}</span>{html.escape(title)}'
        "</div>"
    )


def section_head_html(step: str, title: str, subtitle: str = "") -> str:
    """Legacy alias kept for compatibility — now renders as a compact eyebrow."""
    return eyebrow_html(str(step), title)


def stepper_html(steps: list) -> str:
    """steps: list of (label, state) where state in {'done','current','todo'}."""
    parts = ['<div class="stepper fade-in">']
    n = len(steps)
    for i, (label, state) in enumerate(steps):
        icon = "✓" if state == "done" else str(i + 1)
        parts.append(
            f'<div class="step {state}"><div class="node">{icon}</div>'
            f'<span class="label">{html.escape(label)}</span></div>'
        )
        if i < n - 1:
            filled = "filled" if state == "done" else ""
            parts.append(f'<div class="connector {filled}"></div>')
    parts.append("</div>")
    return "".join(parts)


def case_info_bar_html(icon: str, color: str, specialty: str, title: str, focus_label: str, focus_text: str) -> str:
    return (
        f'<div class="case-info-bar fade-in" style="--case-color:{color};">'
        f'<span class="case-info-icon">{icon}</span>'
        f'<span class="case-specialty">{html.escape(specialty)}</span>'
        f'<span class="case-info-title">{html.escape(title)}</span>'
        f'<span class="case-info-focus">🎯 {html.escape(focus_label)}: {html.escape(focus_text)}</span>'
        "</div>"
    )


def reward_success_html(amount_text: str, balance_label: str, balance_text: str) -> str:
    return (
        '<div class="reward-pop">'
        '<div class="reward-coin">💠</div>'
        f'<div class="reward-amount">+{html.escape(amount_text)}</div>'
        f'<div class="reward-balance">{html.escape(balance_label)}: <b>{html.escape(balance_text)}</b></div>'
        "</div>"
    )


def live_chip_html(text: str) -> str:
    return f'<span class="live-chip"><span class="live-dot"></span>{html.escape(text)}</span>'


def usdt_chip_html(text: str) -> str:
    return f'<span class="usdt-chip">💰 {html.escape(text)}</span>'


def wallet_panel_html(balance: float, lang: str, compact: bool = False) -> str:
    """Yellow-highlighted wallet block for sidebar / USDT zone."""
    from lib.i18n import t

    if compact:
        return (
            f'<div class="usdt-zone wallet-compact">'
            f'<span class="wallet-compact-left">'
            f'<span class="wallet-compact-icon">💰</span>'
            f'<span class="wallet-compact-label">{html.escape(t("sidebar.wallet", lang))}</span>'
            f"</span>"
            f'<span class="wallet-compact-bal">{balance:.2f} <span>USDT</span></span>'
            f"</div>"
        )
    return (
        f'<div class="usdt-zone wallet-panel">'
        f'<div class="usdt-zone-head">'
        f'<span class="usdt-zone-icon">💰</span>'
        f'<span class="usdt-zone-title">{html.escape(t("sidebar.wallet", lang))}</span>'
        f"</div>"
        f'<div class="usdt-balance">{balance:.2f} <span class="usdt-currency">USDT</span></div>'
        f'<div class="usdt-zone-caption">{html.escape(t("sidebar.wallet_caption", lang))}</div>'
        f"</div>"
    )
