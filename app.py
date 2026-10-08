import re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - handled at runtime
    try:
        import fitz
    except ImportError:  # pragma: no cover - handled at runtime
        fitz = None

try:
    from pdf2image import convert_from_bytes
except ImportError:  # pragma: no cover - handled at runtime
    convert_from_bytes = None


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Semester GPA Calculator",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        --background-color: #0b1020;
        --secondary-background-color: #151b2e;
        --text-color: #e5e7eb;
        --primary-color: #6366f1;
        background:
            radial-gradient(ellipse at 5% 0%, rgba(79, 70, 229, .22), transparent 38%),
            radial-gradient(ellipse at 95% 12%, rgba(14, 165, 233, .17), transparent 34%),
            #0b1020;
        color: #e5e7eb;
    }
    .stApp, .stApp [data-testid="stAppViewContainer"] {
        background-color: #0b1020;
    }
    .stApp [data-testid="stHeader"] {
        background: rgba(11, 16, 32, .88);
    }
    .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp [data-testid="stMarkdownContainer"] {
        color: #e5e7eb;
    }
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(120deg, #312e81 0%, #4f46e5 55%, #0284c7 100%);
        box-shadow: 0 18px 45px rgba(49, 46, 129, .2);
    }
    .hero h1 {
        margin: 0 0 .45rem;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .hero p {
        margin: 0;
        color: rgba(255, 255, 255, .85);
        font-size: 1.05rem;
    }
    div[data-testid="stMetric"] {
        padding: 1.25rem 1.4rem;
        background: #151b2e;
        border: 1px solid rgba(99, 102, 241, .14);
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, .24);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--primary-color);
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text-color);
        font-weight: 800;
    }
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border: 1px solid rgba(99, 102, 241, .18);
        border-radius: 14px;
        overflow: hidden;
    }
    div.stButton > button {
        min-height: 3.2rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #4f46e5, #0284c7);
        color: white;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(79, 70, 229, .2);
    }
    div.stButton > button:hover {
        border: 0;
        color: white;
        filter: brightness(1.08);
    }
    .app-footer {
        margin-top: 3rem;
        padding: 1.2rem 0 .4rem;
        border-top: 1px solid rgba(99, 102, 241, .2);
        color: var(--text-color, #334155);
        text-align: center;
        font-size: .9rem;
        opacity: .78;
    }
    </style>
    <section class="hero">
        <h1>🎓 Semester GPA Calculator</h1>
        <p>Upload your marksheet, review the subjects, and calculate your semester results.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------

@st.cache_resource
def get_ocr_reader():
    """
    Create the OCR reader once and reuse it.

    The first run downloads the EasyOCR model.
    """
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "EasyOCR is not installed. Install the app dependencies "
            "with 'python -m pip install -r requirements.txt'."
        ) from exc

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def read_image(image):
    """Read text from one or more uploaded mark-sheet images."""

    reader = get_ocr_reader()

    if not isinstance(image, list):
        image = [image]

    text = []

    for page in image:
        image_array = np.array(page)
        results = reader.readtext(image_array)
        detections = [
            (box, detected_text)
            for box, detected_text, confidence in results
            if confidence >= 0.30
        ]
        text.extend(group_ocr_detections(detections))

    return text


def group_ocr_detections(detections):
    """Join OCR text boxes that share a visual row, in left-to-right order."""

    if not detections:
        return []

    prepared = []
    heights = []

    for box, detected_text in detections:
        top = min(point[1] for point in box)
        bottom = max(point[1] for point in box)
        center_y = (top + bottom) / 2
        height = bottom - top
        heights.append(height)
        prepared.append((center_y, top, bottom, height, box, detected_text))

    row_tolerance = max(10.0, float(np.median(heights)) * 0.65)
    prepared.sort(key=lambda item: item[0])
    grouped = []

    for center_y, top, bottom, height, box, detected_text in prepared:
        matching_group = None
        for group in reversed(grouped):
            group_center = group["center"]
            if center_y - group_center > row_tolerance:
                break
            if abs(center_y - group_center) <= row_tolerance:
                matching_group = group
                break

        item = (min(point[0] for point in box), detected_text)

        if matching_group is None:
            grouped.append({
                "center": center_y,
                "top": top,
                "bottom": bottom,
                "items": [item],
            })
        else:
            matching_group["items"].append(item)
            matching_group["center"] = (
                matching_group["center"] + center_y
            ) / 2
            matching_group["top"] = min(matching_group["top"], top)
            matching_group["bottom"] = max(matching_group["bottom"], bottom)

    return [
        " ".join(text for _, text in sorted(group["items"]))
        for group in grouped
    ]


def convert_pdf_to_images(uploaded_file):
    """Render each PDF page into an image so OCR can read it."""

    pdf_bytes = uploaded_file.getvalue()

    if fitz is not None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(image)

        if not pages:
            raise RuntimeError("No pages were found in the uploaded PDF.")

        return pages

    if convert_from_bytes is not None:
        pages = convert_from_bytes(pdf_bytes, dpi=200)

        if not pages:
            raise RuntimeError("No pages were found in the uploaded PDF.")

        return pages

    raise RuntimeError(
        "PDF support requires a PDF rendering library. "
        "Install dependencies with 'python -m pip install -r requirements.txt'."
    )


# ---------------------------------------------------------
# Mark-sheet parser
# ---------------------------------------------------------

def parse_line(line, credit_point_layout=False):
    """
    Try to identify:

        Subject   Credits   Grade Point   Credit Point

    Example:

        Mathematics  4  8  32

    This parser is intentionally simple.
    It can be customized for a specific university
    mark-sheet format later.
    """

    # Find numbers such as:
    # 3
    # 4
    # 8
    # 7.5
    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        line,
    )

    # The review table is only for rows with numeric marks data, not arbitrary
    # OCR text such as headings, names, dates, or footnotes.
    if len(numbers) < 2:
        return None

    values = [float(number) for number in numbers]

    # Remove numbers from the line to obtain the subject.
    subject = re.sub(
        r"\d+(?:\.\d+)?",
        "",
        line,
    )
    subject = re.sub(
        r"\b(pass|fail|th\.?|pr\.?)\b",
        "",
        subject,
        flags=re.IGNORECASE,
    )

    subject = re.sub(
        r"\s+",
        " ",
        subject,
    ).strip(" -:|")

    if not subject or not re.search(r"[a-zA-Z]", subject):
        return None

    if re.search(
        r"\b(total|semester|gpa|cgpa|percentage|credits?|grade|result|marks?|"
        r"subject|name|hours?|points?|serial|si|no|student|father|mother|"
        r"home|university|examinations?|announced|date|register|reg|"
        r"click|view|details|print|note|provisional|responsible|published|"
        r"net|grievances?|communicated|nodal|officer|college)\b",
        subject,
        flags=re.IGNORECASE,
    ):
        return None

    credits = None
    grade_point = None

    if credit_point_layout and len(values) == 2:
        # Some mark sheets print Grade Point and Credit Point only.
        grade_point, credit_points = values
        if grade_point > 0:
            credits = credit_points / grade_point
        elif credit_points == 0:
            credits = None
    else:
        # Standard rows list Credits, Grade Point, then optionally Credit Points.
        credit_index = next(
            (
                index
                for index, value in enumerate(values)
                if 0 < value <= 10
            ),
            None,
        )
        if credit_index is not None:
            credits = values[credit_index]
            grade_point = next(
                (
                    value
                    for value in values[credit_index + 1:]
                    if 0 <= value <= 10
                ),
                None,
            )

    return {
        "Subject": subject,
        "Credits": credits,
        "Grade Point": grade_point,
    }


def parse_ocr_text(lines):
    """Convert OCR lines into subject rows, retaining partial detections."""

    document_text = " ".join(lines).casefold()
    credit_point_layout = (
        ("credit point" in document_text and "grade point" in document_text)
        or (
            document_text.count("credit") >= 2
            and document_text.count("grade") >= 2
            and "hours" in document_text
        )
    )

    rows = []

    for line in lines:
        row = parse_line(line, credit_point_layout=credit_point_layout)
        if row is not None:
            rows.append(row)

    unique_rows = []
    seen = set()
    for row in rows:
        key = row["Subject"].casefold()
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    return unique_rows


# ---------------------------------------------------------
# Calculations
# ---------------------------------------------------------

def calculate_gpa(df):
    """
    GPA =
        sum(Credits × Grade Point)
        /
        sum(Credits)
    """

    if df.empty:
        return 0.0

    credits = pd.to_numeric(
        df["Credits"],
        errors="coerce",
    )

    grade_points = pd.to_numeric(
        df["Grade Point"],
        errors="coerce",
    )

    if credits.isna().any() or grade_points.isna().any():
        raise ValueError("Every subject must have credits and a grade point.")

    if (credits < 0).any() or (credits > 10).any():
        raise ValueError("Credits must be between 0 and 10.")

    if (grade_points < 0).any() or (grade_points > 10).any():
        raise ValueError("Grade points must be between 0 and 10.")

    total_credits = credits.sum()

    if total_credits == 0:
        raise ValueError("Total credits cannot be zero.")

    total_credit_points = (
        credits * grade_points
    ).sum()

    return total_credit_points / total_credits


def calculate_percentage(cgpa):
    """Convert the semester GPA to percentage and clamp to zero minimum."""

    return max(0.0, (float(cgpa) - 0.5) * 10)


def calculate_results(df):
    """Return computed metrics for a subject table or an error message."""

    if df.empty:
        return None, "Please add at least one subject."

    cleaned = df.copy()
    cleaned["Subject"] = cleaned["Subject"].fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned["Subject"] != ""]

    if cleaned.empty:
        return None, "Please add at least one subject."

    cleaned["Credits"] = pd.to_numeric(cleaned["Credits"], errors="coerce")
    cleaned["Grade Point"] = pd.to_numeric(
        cleaned["Grade Point"],
        errors="coerce",
    )

    if cleaned[["Credits", "Grade Point"]].isna().any().any():
        return None, (
            "Each subject row needs both credits and a grade point. "
            "Please correct the highlighted or empty values."
        )

    try:
        gpa = calculate_gpa(cleaned)
    except ValueError as exc:
        return None, str(exc)

    percentage = calculate_percentage(gpa)

    return {
        "GPA": gpa,
        "Percentage": percentage,
    }, None


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload mark sheet",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "pdf",
    ],
)


if uploaded_file:

    file_name = uploaded_file.name.lower()
    is_pdf = file_name.endswith(".pdf") or uploaded_file.type == "application/pdf"

    if is_pdf:
        try:
            images = convert_pdf_to_images(uploaded_file)
            preview_image = images[0]
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    else:
        preview_image = Image.open(uploaded_file)
        images = [preview_image]

    st.image(
        preview_image,
        caption="Uploaded mark sheet",
        width="stretch",
    )

    if st.button(
        "🔍 Read Mark Sheet",
        type="primary",
        width="stretch",
    ):

        with st.spinner(
            "Reading mark sheet..."
        ):

            try:
                ocr_lines = read_image(images)
                rows = parse_ocr_text(
                    ocr_lines
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

            st.session_state["ocr_lines"] = ocr_lines
            st.session_state["rows"] = rows


# ---------------------------------------------------------
# Display OCR text
# ---------------------------------------------------------

if "ocr_lines" in st.session_state:

    with st.expander("View detected text"):

        for line in st.session_state["ocr_lines"]:
            st.write(line)


# ---------------------------------------------------------
# Subject table
# ---------------------------------------------------------

if "rows" in st.session_state:

    st.subheader("Review subjects")

    if st.session_state["rows"]:
        st.caption(
            "Detected subjects are listed below. Review each row and fill "
            "any blank credits or grade points before calculating."
        )
    elif st.session_state.get("ocr_lines"):
        st.warning(
            "Text was detected, but no subject rows could be identified. "
            "The mark sheet should show each subject with its credits and grade point."
        )

    df = pd.DataFrame(
        st.session_state["rows"],
        columns=[
            "Subject",
            "Credits",
            "Grade Point",
        ],
    )

else:

    # Allow completely manual entry.
    df = pd.DataFrame(
        columns=[
            "Subject",
            "Credits",
            "Grade Point",
        ]
    )


edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    width="stretch",
    column_config={
        "Subject": st.column_config.TextColumn(
            "Subject",
        ),
        "Credits": st.column_config.NumberColumn(
            "Credits",
            min_value=0,
            max_value=10,
            step=0.5,
        ),
        "Grade Point": st.column_config.NumberColumn(
            "Grade Point",
            min_value=0,
            max_value=10,
            step=0.01,
        ),
    },
)


# ---------------------------------------------------------
# Calculate results
# ---------------------------------------------------------

if st.button(
    "🧮 Calculate GPA & Percentage",
    type="primary",
    width="stretch",
):
    results, error_message = calculate_results(edited_df)

    if error_message:
        st.warning(error_message)
    else:
        st.divider()

        st.subheader("Semester Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "GPA",
                f"{results['GPA']:.2f}",
            )

        with col2:
            st.metric(
                "Percentage",
                f"{results['Percentage']:.1f}%",
            )



st.markdown(
    '<footer class="app-footer">Developed by Mohamed Imraan</footer>',
    unsafe_allow_html=True,
)