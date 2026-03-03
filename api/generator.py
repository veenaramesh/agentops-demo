import io
import zipfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Undefined

TEMPLATE_DIR = Path(__file__).parent.parent / "template"


def generate_project_zip(config: dict) -> io.BytesIO:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    buf = io.BytesIO()
    project = config["project_name"]

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(TEMPLATE_DIR.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(TEMPLATE_DIR)
            dest = Path(project) / rel

            if src.suffix == ".j2":
                # Render template; empty result (opted-out bundle) → skip
                rendered = env.get_template(str(rel)).render(**config).strip()
                if rendered:
                    dest = dest.with_suffix("")  # strip .j2
                    zf.writestr(str(dest), rendered + "\n")
            else:
                zf.writestr(str(dest), src.read_bytes())

    buf.seek(0)
    return buf
