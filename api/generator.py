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

            # tool_stub.py.j2 is handled by the per-tool loop below
            if "tool_stub.py.j2" in str(rel):
                continue

            dest = Path(project) / rel

            if src.suffix == ".j2":
                # Render template; empty result (opted-out bundle) → skip
                rendered = env.get_template(str(rel)).render(**config).strip()
                if rendered:
                    dest = dest.with_suffix("")  # strip .j2
                    zf.writestr(str(dest), rendered + "\n")
            else:
                zf.writestr(str(dest), src.read_bytes())

        # Per-tool stub files (one .py per UC function node)
        if config.get("include_tools") == "yes":
            stub_tmpl = env.get_template("tools/agent_tools/tool_stub.py.j2")
            for tool in config.get("tools", []):
                rendered = stub_tmpl.render(tool=tool, **config).strip()
                dest = Path(project) / "tools" / "agent_tools" / f"{tool['name']}.py"
                zf.writestr(str(dest), rendered + "\n")

    buf.seek(0)
    return buf
