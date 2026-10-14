"""Restrained banking-analytics presentation, without external assets."""

CSS = """
<style>
  :root { --navy:#132C45; --teal:#137F89; --slate:#587086; }
  .stApp { background:#F6F8FB; }
  .block-container { max-width:1320px; padding-top:4.8rem; padding-bottom:3rem; }
  [data-testid="stSidebar"] { background:#132C45; }
  [data-testid="stSidebar"] * { color:#F0F5F9; }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#AFC1D0; }
  [data-testid="stSidebar"] a, [data-testid="stSidebar"] a * { color:#B7E4E3 !important; }
  [data-testid="stSidebar"] [role="radiogroup"] { gap:.28rem; }
  [data-testid="stSidebar"] label { padding:.42rem 0; font-size:.86rem; }
  h1,h2,h3 { letter-spacing:-.025em; color:#132C45; }
  h1 { font-size:2.6rem !important; font-weight:720 !important; }
  h2 { font-size:1.52rem !important; margin-top:.4rem; }
  h3 { font-size:1.14rem !important; }
  .eyebrow { color:#137F89; font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:0 0 .45rem; }
  .hero-intro { color:#587086; font-size:1.06rem; line-height:1.65; max-width:850px; margin:0 0 1.5rem; }
  .sidebar-brand { font-size:1.7rem; font-weight:750; letter-spacing:-.04em; margin-bottom:.3rem; }
  .sidebar-brand span { color:#78CFCA !important; }
  .sidebar-label { font-size:.75rem; letter-spacing:.1em; text-transform:uppercase; color:#AFC1D0 !important; margin-top:2rem; }
  [data-testid="stMetric"] { background:white; border:1px solid #DEE6EE; border-radius:10px; padding:1rem 1.05rem; min-height:110px; }
  [data-testid="stMetricLabel"] { color:#587086; font-size:.8rem; }
  [data-testid="stMetricValue"] { color:#132C45; font-size:1.85rem; }
  [data-testid="stExpander"] { background:#FFFFFF; border-color:#DEE6EE; border-radius:9px; }
  [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#DEE6EE !important; border-radius:10px; }
  [data-testid="stDataFrame"] { border:1px solid #DEE6EE; border-radius:8px; }
  [data-testid="stPlotlyChart"] { background:white; border-radius:10px; }
  a { color:#137F89; }
  .context-note { border-left:3px solid #137F89; padding:.7rem 1rem; background:#EAF3F4; color:#294D62; font-size:.87rem; line-height:1.6; margin:.7rem 0 1.3rem; border-radius:0 7px 7px 0; }
  .app-footer { color:#587086; font-size:.76rem; border-top:1px solid #DEE6EE; padding-top:1rem; margin-top:2rem; line-height:1.7; }
  @media(max-width:700px) { .block-container { padding-left:1rem; padding-right:1rem; } h1 { font-size:2.05rem !important; } }
  @media(prefers-reduced-motion:reduce) { *,*:before,*:after { scroll-behavior:auto !important; transition:none !important; animation:none !important; } }
</style>
"""
