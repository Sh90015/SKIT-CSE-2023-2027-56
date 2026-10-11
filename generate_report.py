#!/usr/bin/env python3
"""Generate weekly, monthly, or final GitHub contribution PDF reports."""

import datetime as dt
import html
import io
import os
import subprocess
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

COLLEGE_NAME = (
    "Swami Keshvanand Institute of Technology, "
    "Management & Gramothan, Jaipur"
)
DEPARTMENT_NAME = "Department of Computer Science & Engineering"


def run_git(args):
    """Run Git and return its output."""
    return subprocess.check_output(
        ["git", *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        stderr=subprocess.PIPE,
    ).strip()


def get_repo_info():
    """Get repository name and current branch."""
    try:
        root = run_git(["rev-parse", "--show-toplevel"])
        repo_name = os.path.basename(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        repo_name = os.path.basename(os.getcwd()) or "Project-Repository"

    try:
        branch = run_git(["branch", "--show-current"]) or "detached-HEAD"
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"

    return repo_name, branch


def get_git_metrics(interval="weekly"):
    """Calculate contribution metrics from Git commit history."""
    if interval not in {"weekly", "monthly", "final"}:
        raise ValueError("Choose weekly, monthly, or final.")

    today = dt.date.today()

    args = [
        "log",
        "--no-merges",
        "--date=short",
        "--pretty=format:COMMIT|||%h|||%an|||%ad|||%s",
        "--numstat",
    ]

    if interval == "weekly":
        since = (today - dt.timedelta(days=7)).isoformat()
        args.append(f"--since={since}")
        scope = f"Last 7 days (since {since})"
    elif interval == "monthly":
        since = (today - dt.timedelta(days=30)).isoformat()
        args.append(f"--since={since}")
        scope = f"Last 30 days (since {since})"
    else:
        scope = "Complete project lifecycle (all available commits)"

    try:
        raw = run_git(args)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            "Could not read Git history. Run this script inside "
            "a Git repository.\n" + str(detail)
        ) from exc

    students = defaultdict(
        lambda: {
            "commits": 0,
            "added": 0,
            "deleted": 0,
            "active_days": set(),
        }
    )
    timeline = defaultdict(lambda: defaultdict(int))
    logs = defaultdict(list)

    current_author = None

    for raw_line in raw.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("COMMIT|||"):
            parts = line.split("|||", 4)

            if len(parts) != 5:
                current_author = None
                continue

            _, sha, author, date_str, message = parts
            author = author.strip()

            # Exclude automated bot accounts.
            if (
                "bot" in author.lower()
                or "github-actions" in author.lower()
            ):
                current_author = None
                continue

            current_author = author
            date_str = date_str.strip()

            students[author]["commits"] += 1
            students[author]["active_days"].add(date_str)
            logs[author].append(
                (date_str, sha.strip(), message.strip())
            )

            try:
                commit_date = dt.datetime.strptime(
                    date_str, "%Y-%m-%d"
                ).date()

                if interval == "weekly":
                    period = commit_date.strftime("%Y-%m-%d")
                elif interval == "monthly":
                    iso = commit_date.isocalendar()
                    period = f"{iso.year}-W{iso.week:02d}"
                else:
                    period = commit_date.strftime("%Y-%m")

                timeline[period][author] += 1

            except ValueError:
                pass

            continue

        # Git numstat format: added<TAB>deleted<TAB>filename.
        # Binary file changes use "-" and are not counted.
        if current_author:
            fields = line.split("\t", 2)

            if (
                len(fields) == 3
                and fields[0].isdigit()
                and fields[1].isdigit()
            ):
                students[current_author]["added"] += int(fields[0])
                students[current_author]["deleted"] += int(fields[1])

    return students, timeline, logs, scope


def create_charts(students, timeline, interval):
    """Generate commit timeline and net-lines-changed charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3))

    authors = sorted(students)
    periods = sorted(timeline)

    if periods and authors:
        for author in authors:
            values = [
                timeline[period].get(author, 0)
                for period in periods
            ]
            ax1.plot(
                periods,
                values,
                marker="o",
                linewidth=1.5,
                label=author,
            )

        ax1.set_title(f"Commit timeline ({interval})")
        ax1.set_ylabel("Commits")
        ax1.tick_params(axis="x", rotation=35, labelsize=7)
        ax1.grid(True, linestyle="--", alpha=0.4)

        if len(authors) <= 8:
            ax1.legend(fontsize=6, loc="best")
    else:
        ax1.text(
            0.5, 0.5, "No commits in this period",
            ha="center", va="center",
        )
        ax1.set_axis_off()

    if authors:
        net = [
            students[a]["added"] - students[a]["deleted"]
            for a in authors
        ]

        ax2.bar(range(len(authors)), net, width=0.65)
        ax2.set_xticks(range(len(authors)))
        ax2.set_xticklabels(
            authors, rotation=35, ha="right", fontsize=7
        )
        ax2.set_title("Net lines changed")
        ax2.set_ylabel("Added - deleted")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        ax2.text(
            0.5, 0.5, "No line changes recorded",
            ha="center", va="center",
        )
        ax2.set_axis_off()

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)
    buffer.seek(0)

    return Image(
        buffer, width=500, height=157, kind="proportional"
    )


def generate_pdf(interval="weekly"):
    """Generate the requested PDF report."""
    students, timeline, student_logs, scope = get_git_metrics(interval)
    repo_name, branch_name = get_repo_info()
    date_stamp = dt.date.today().isoformat()

    if interval == "weekly":
        title = "Weekly Progress Report (Form-3)"
        filename = (
            f"{repo_name}_Weekly_Progress_Report_Form-3_"
            f"{date_stamp}.pdf"
        )
    elif interval == "monthly":
        title = "Monthly Progress Report (Form-3)"
        filename = (
            f"{repo_name}_Monthly_Progress_Report_Form-3_"
            f"{date_stamp}.pdf"
        )
    else:
        title = "Final Project Evaluation Report"
        filename = f"{repo_name}_Final_Report_{date_stamp}.pdf"

    styles = getSampleStyleSheet()

    college_style = ParagraphStyle(
        "College",
        parent=styles["Heading1"],
        fontSize=12,
        leading=15,
        alignment=1,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    )

    dept_style = ParagraphStyle(
        "Department",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#475569"),
        spaceAfter=5,
    )

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        alignment=1,
        textColor=colors.HexColor("#1D4ED8"),
        spaceAfter=6,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=10,
        leading=13,
        spaceBefore=7,
        spaceAfter=4,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
    )

    cell_style = ParagraphStyle(
        "Cell",
        parent=small_style,
        wordWrap="CJK",
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=28,
        bottomMargin=28,
    )

    story = [
        Paragraph(
            f"<b>{html.escape(COLLEGE_NAME)}</b>",
            college_style,
        ),
        Paragraph(
            f"<b>{html.escape(DEPARTMENT_NAME)}</b>",
            dept_style,
        ),
        Paragraph(
            f"<u><b>{html.escape(title)}</b></u>",
            title_style,
        ),
        Paragraph(
            f"<b>Repository:</b> {html.escape(repo_name)}"
            f" &nbsp; | &nbsp; "
            f"<b>Branch:</b> {html.escape(branch_name)}",
            styles["Normal"],
        ),
        Paragraph(
            f"<b>Evaluation window:</b> {html.escape(scope)}"
            f" &nbsp; | &nbsp; "
            f"<b>Generated:</b> "
            f"{dt.date.today().strftime('%B %d, %Y')}",
            small_style,
        ),
        Spacer(1, 6),
        Paragraph(
            "1. Individual Contribution Breakdown",
            section_style,
        ),
    ]

    total_commits = sum(
        item["commits"] for item in students.values()
    )

    summary = [[
        "Student / Author",
        "Commits (%)",
        "Added",
        "Deleted",
        "Net LOC",
        "Active days",
    ]]

    if students:
        for name in sorted(students):
            data = students[name]
            percentage = (
                data["commits"] * 100 / total_commits
                if total_commits else 0
            )

            summary.append([
                Paragraph(html.escape(name), cell_style),
                f"{data['commits']} ({percentage:.1f}%)",
                f"+{data['added']:,}",
                f"-{data['deleted']:,}",
                f"{data['added'] - data['deleted']:,}",
                str(len(data["active_days"])),
            ])
    else:
        summary.append([
            Paragraph("No commits found in this period.", cell_style),
            "-", "-", "-", "-", "-",
        ])

    summary_table = Table(
        summary,
        colWidths=[135, 82, 66, 66, 66, 65],
        repeatRows=1,
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F8FAFC")
        ]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    story.extend([
        summary_table,
        Spacer(1, 5),
        Paragraph("2. Visual Trends & Volume", section_style),
        create_charts(students, timeline, interval),
        Spacer(1, 4),
        Paragraph(
            "3. Detailed Commit Logs & Mentor Evaluation",
            section_style,
        ),
    ])

    if not student_logs:
        story.append(Paragraph(
            "No commit logs found for this timeframe.",
            styles["Normal"],
        ))
    else:
        for student_name in sorted(student_logs):
            logs = student_logs[student_name]

            story.append(Paragraph(
                f"<b>Student / Author:</b> "
                f"{html.escape(student_name)} — "
                f"{len(logs)} commit(s)",
                styles["Heading3"],
            ))

            rows = [[
                "Date", "Hash", "Commit message", "Mentor marks (/10)"
            ]]

            for index, (date_value, sha, message) in enumerate(logs):
                rows.append([
                    date_value,
                    Paragraph(html.escape(sha), cell_style),
                    Paragraph(
                        html.escape(message or "(No commit message)"),
                        cell_style,
                    ),
                    Paragraph(
                        "<b>_____ / 10</b>" if index == 0 else "",
                        cell_style,
                    ),
                ])

            log_table = Table(
                rows,
                colWidths=[58, 55, 295, 72],
                repeatRows=1,
                splitByRow=1,
            )

            table_style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (3, 1), (3, -1), colors.HexColor("#FEF3C7")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]

            if len(rows) > 2:
                table_style.extend([
                    ("SPAN", (3, 1), (3, len(rows) - 1)),
                    ("VALIGN", (3, 1), (3, len(rows) - 1), "MIDDLE"),
                ])

            log_table.setStyle(TableStyle(table_style))
            story.extend([log_table, Spacer(1, 5)])

    mentor = Paragraph(
        "<b>Project Mentor</b><br/><br/>"
        "Name: ____________________<br/><br/>"
        "Designation: Project Mentor<br/><br/>"
        "Signature: ____________________",
        small_style,
    )

    coordinator = Paragraph(
        "<b>Lab Coordinator</b><br/><br/>"
        "Name: ____________________<br/><br/>"
        "Designation: Lab Coordinator<br/><br/>"
        "Signature: ____________________",
        small_style,
    )

    signature_table = Table(
        [[mentor, coordinator]],
        colWidths=[270, 270],
    )

    signature_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    story.extend([Spacer(1, 10), signature_table])
    doc.build(story)

    print(f"[SUCCESS] Generated: {filename}")
    print(
        f"Found {len(students)} author(s) and "
        f"{total_commits} commit(s)."
    )


if __name__ == "__main__":
    selected = (
        sys.argv[1].lower() if len(sys.argv) > 1 else "weekly"
    )

    try:
        generate_pdf(selected)
    except (ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)