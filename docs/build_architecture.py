"""Build the self-contained animated architecture HTML (auto-play and presenter versions).

Inlines the official Databricks product icons (SVG) into architecture_template.html,
plus a few simple custom icons drawn in the same lava/peach style for things that
have no official icon (model, app, vendors, outcomes, profiling).

Usage: python3 build_architecture.py [icon_dir]
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
ICON_DIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "Downloads")

OFFICIAL = {
    "DATABRICKS": "databricks-symbol-color.svg",
    "I_UC": "unity-catalog-icon-full-color.svg",
    "I_LAKEBASE": "lakebase-icon-full-color.svg",
    "I_JOBS": "lakeflow-jobs-icon-full-color.svg",
    "I_DELTA": "primary-icon-orange-delta-table.svg",
    "I_DASH": "ai-bi-dashboards-icon-full-color.svg",
    "I_SERVERLESS": "primary-icon-orange-serverless.svg",
}

L, P = "#FF5F46", "#FABFBA"
CUSTOM = {
    # model: small network of nodes
    "I_MODEL": f'<svg viewBox="0 0 512 512" fill="none"><path d="M136 136L256 256M376 136L256 256M256 256L136 376M256 256L376 376" stroke="{P}" stroke-width="28"/>'
               f'<circle cx="256" cy="256" r="62" fill="{L}"/><circle cx="120" cy="120" r="46" fill="{P}"/><circle cx="392" cy="120" r="46" fill="{P}"/>'
               f'<circle cx="120" cy="392" r="46" fill="{L}"/><circle cx="392" cy="392" r="46" fill="{L}"/></svg>',
    # generic operational database
    "I_DB": f'<svg viewBox="0 0 512 512" fill="none"><ellipse cx="256" cy="120" rx="160" ry="56" fill="{L}"/>'
            f'<path d="M96 120V392C96 423 168 448 256 448C344 448 416 423 416 392V120C416 151 344 176 256 176C168 176 96 151 96 120Z" fill="{P}"/>'
            f'<path d="M96 256C96 287 168 312 256 312C344 312 416 287 416 256" stroke="{L}" stroke-width="24"/></svg>',
    # loan application: phone with form lines
    "I_APP": f'<svg viewBox="0 0 512 512" fill="none"><rect x="136" y="48" width="240" height="416" rx="40" fill="{P}"/>'
             f'<rect x="168" y="104" width="176" height="272" rx="12" fill="#fff"/><rect x="192" y="140" width="128" height="22" rx="11" fill="{L}"/>'
             f'<rect x="192" y="190" width="96" height="22" rx="11" fill="{P}"/><rect x="192" y="240" width="128" height="22" rx="11" fill="{P}"/>'
             f'<rect x="192" y="310" width="128" height="40" rx="12" fill="{L}"/><circle cx="256" cy="420" r="18" fill="#fff"/></svg>',
    # vendors: bank building
    "I_BANK": f'<svg viewBox="0 0 512 512" fill="none"><path d="M256 48L464 160H48L256 48Z" fill="{L}"/>'
              f'<rect x="88" y="184" width="48" height="200" fill="{P}"/><rect x="180" y="184" width="48" height="200" fill="{P}"/>'
              f'<rect x="284" y="184" width="48" height="200" fill="{P}"/><rect x="376" y="184" width="48" height="200" fill="{P}"/>'
              f'<rect x="48" y="400" width="416" height="56" fill="{L}"/></svg>',
    # outcomes: checklist
    "I_CHECK": f'<svg viewBox="0 0 512 512" fill="none"><rect x="96" y="56" width="320" height="400" rx="32" fill="{P}"/>'
               f'<path d="M150 170L190 210L262 138" stroke="{L}" stroke-width="30" stroke-linecap="round" stroke-linejoin="round"/>'
               f'<rect x="290" y="160" width="80" height="26" rx="13" fill="#fff"/>'
               f'<path d="M160 300L220 360M220 300L160 360" stroke="{L}" stroke-width="30" stroke-linecap="round"/>'
               f'<rect x="290" y="318" width="80" height="26" rx="13" fill="#fff"/></svg>',
    # data profiling: pulse line over a baseline
    "I_PULSE": f'<svg viewBox="0 0 512 512" fill="none"><rect x="48" y="80" width="416" height="352" rx="36" fill="{P}"/>'
               f'<path d="M96 300H176L216 184L272 364L320 240L352 300H416" stroke="{L}" stroke-width="30" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    # envelope (request / response)
    "I_ENV": '<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="2" stroke="#fff" stroke-width="2"/>'
             '<path d="M4 7l8 6 8-6" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
}


def clean(svg: str) -> str:
    svg = re.sub(r"<\?xml[^>]*>", "", svg).strip()
    return re.sub(r'\s(width|height)="[^"]*"', "", svg, count=2)


def main():
    icons = {k: clean((ICON_DIR / f).read_text()) for k, f in OFFICIAL.items()}
    icons.update(CUSTOM)
    # white serverless icon for the dark endpoint header
    icons["I_SERVERLESS_W"] = icons["I_SERVERLESS"].replace("#FF5F46", "#FFFFFF")
    html = (HERE / "architecture_template.html").read_text()
    for k, v in icons.items():
        html = html.replace("{{" + k + "}}", v)
    # two outputs: timed loop for recording, and a keyboard-driven version for presenting live
    for mode, name in [("auto", "architecture.html"), ("present", "architecture_presenter.html")]:
        page = html.replace("{{MODE}}", mode)
        left = re.findall(r"\{\{[A-Z_]+\}\}", page)
        if left:
            raise SystemExit(f"unreplaced placeholders: {sorted(set(left))}")
        out = HERE / name
        out.write_text(page)
        print(out)


if __name__ == "__main__":
    main()
