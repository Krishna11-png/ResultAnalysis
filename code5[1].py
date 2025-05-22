import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
import plotly.express as px

st.set_page_config(page_title="CSE Result Analysis", layout="wide")

# --- Custom CSS for white headings, white tabs, black download button text ---
st.markdown("""
<style>
body, .main, [data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #0a1f4c 0%, #051747 100%) !important;
    color: white !important;
}
h1, h2, h3, h4, h5, h6, .stMetric, .stDataFrame, .stExpander, .stSelectbox label, .stSlider label, .stMarkdown, .stTable, .stDataFrame th, .stDataFrame td, .stTabs label, .stTabs span {
    color: #fff !important;
}
.stDataFrame th, .stDataFrame td {
    background-color: #1a2c5b !important;
    color: #fff !important;
}
/* Ribbon/tabs white background and dark text */
.stTabs [data-baseweb="tab-list"] {
    background: #fff !important;
}
.stTabs [data-baseweb="tab"] {
    color: #051747 !important;
    font-weight: bold;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1a2c5b !important;
}
/* Download button white background, black text */
.stButton>button, .stDownloadButton>button {
    background-color: #fff !important;
    color: #051747 !important;
    font-weight: bold;
    border: 2px solid #051747 !important;
}
</style>
""", unsafe_allow_html=True)

# --- NIET LOGO at the top ---
st.image("https://pplx-res.cloudinary.com/image/private/user_uploads/50098909/fJUmtInfVAnTyJL/logo.jpg", width=300)

st.title("GradeVista:Academic Result Analysis System")

def sanitize_sheet_name(name):
    return re.sub(r'[\[\]\:\*\?\/\\]', '_', name)[:31]

def flatten_headers(df):
    df.columns = [' '.join([str(i) for i in col if str(i) != 'nan']).strip() for col in df.columns.values]
    return df

def extract_subject_blocks(df):
    subject_blocks = []
    processed = set()
    for col in df.columns:
        if 'Internal' in col:
            subj = col.split('Internal')[0].strip()
            # Exclude Cyber Security from analysis
            if subj in processed or 'cyber security' in subj.lower():
                continue
            external_col = None
            total_col = None
            for other_col in df.columns:
                if f"{subj} External" in other_col:
                    external_col = other_col
                if f"{subj} Total" in other_col:
                    total_col = other_col
            if external_col and total_col:
                subject_blocks.append({
                    'name': subj,
                    'internal': col,
                    'external': external_col,
                    'total': total_col
                })
                processed.add(subj)
    return subject_blocks

def clean_numeric(column):
    def fix(val):
        if isinstance(val, str):
            match = re.search(r'\d+', val)
            if match:
                return int(match.group(0))
            return np.nan
        return val
    return pd.to_numeric(column.apply(fix), errors='coerce')

def is_lab_subject(subject_name):
    lab_keywords = ['Lab', 'Laboratory', 'Practical']
    return any(kw.lower() in subject_name.lower() for kw in lab_keywords)

def analyze_subject(df, block, roll_col, name_col):
    internal = clean_numeric(df[block['internal']])
    external = clean_numeric(df[block['external']])
    total = clean_numeric(df[block['total']])
    valid_mask = df[roll_col].notnull() & (df[roll_col].astype(str).str.strip() != '') & external.notna()
    appeared = valid_mask.sum()
    if appeared == 0:
        return None, None
    if is_lab_subject(block['name']):
        pass_mask = valid_mask  # All labs are passed
        ranges = {
            '90-100': ((external >= 90) & (external <= 100) & valid_mask).sum(),
            '80-89': ((external >= 80) & (external < 90) & valid_mask).sum(),
            '70-79': ((external >= 70) & (external < 80) & valid_mask).sum(),
            '60-69': ((external >= 60) & (external < 70) & valid_mask).sum(),
            '50-59': ((external >= 50) & (external < 60) & valid_mask).sum(),
            '<50': ((external < 50) & valid_mask).sum()
        }
    else:
        pass_mask = (internal >= 20) & (external >= 40) & (total >= 60) & valid_mask
        ranges = {
            '40-50': ((external >= 40) & (external <= 50) & valid_mask).sum(),
            '51-60': ((external >= 51) & (external <= 60) & valid_mask).sum(),
            '61-70': ((external >= 61) & (external <= 70) & valid_mask).sum(),
            '71-80': ((external >= 71) & (external <= 80) & valid_mask).sum(),
            '>81': ((external > 81) & valid_mask).sum()
        }
    passed = pass_mask.sum()
    failed = appeared - passed
    subject_result = {
        'Subject Name': block['name'],
        'Total Students': appeared,
        'Passed Students': passed,
        'Failed Students': failed,
        'Pass Percentage': round(100 * passed / appeared, 2) if appeared > 0 else 0,
        'Highest Marks': external[valid_mask].max(),
        'Average Marks': round(external[valid_mask].mean(), 2),
        **ranges
    }
    failed_students = df.loc[~pass_mask & valid_mask, :]
    failed_students = failed_students[failed_students[roll_col].notnull()]
    failed_students = failed_students[failed_students[roll_col].astype(str).str.strip() != '']
    if not failed_students.empty:
        failed_students = failed_students.reset_index(drop=True)
        failed_students['Sr. No.'] = failed_students.index + 1
        result = pd.DataFrame({
            'Sr. No.': failed_students['Sr. No.'],
            'Roll No.': failed_students[roll_col],
            'Name of Students': failed_students[name_col],
            'Section': failed_students['Section'],
            'Type of Student': 'Day Scholar',
            'Internal': failed_students[block['internal']],
            'External': failed_students[block['external']],
            'Total': failed_students[block['total']],
            'Sessional 1': '',
            'Sessional 2': '',
            'Sessional 3': '',
            'Attendance in Subject': '',
            'Overall Attendance': '',
            'Mentor Remark': '',
            'Faculty Remark': '',
            'HOD Remark': ''
        })
    else:
        result = pd.DataFrame()
    return subject_result, result

def generate_semester_summary(df, subject_blocks, roll_col):
    # Hardcoded as per your request
    summary = pd.DataFrame([{
        'Section': 'A+B+C+D',
        'Total Students': 290,
        'No. Students Passed': 270,
        'No. Students Failed': 20,
        'Overall % Result of Semester': 93,
        'No of Students With 1 Backlog': 12,
        'No of Students With 2 Backlog': 5,
        'No of Students With more than 2 Backlog': 3
    }])
    return summary

def top_performers_overall(df, subject_blocks, roll_col, name_col):
    total_marks = {}
    for idx, row in df.iterrows():
        roll = str(row[roll_col]).strip()
        if roll == '':
            continue
        total = 0
        for block in subject_blocks:
            val = row[block['total']]
            val_num = pd.to_numeric(val, errors='coerce')
            if not pd.isna(val_num):
                total += val_num
        total_marks[roll] = (row[name_col], total)
    top5 = sorted(total_marks.items(), key=lambda x: x[1][1], reverse=True)[:5]
    return pd.DataFrame([
        {'Roll No.': roll, 'Name': name, 'Total Marks': marks}
        for roll, (name, marks) in top5
    ])

def top_performers_subject(df, block, roll_col, name_col):
    marks = []
    for idx, row in df.iterrows():
        roll = str(row[roll_col]).strip()
        if roll == '':
            continue
        val = row[block['total']]
        val_num = pd.to_numeric(val, errors='coerce')
        if not pd.isna(val_num):
            marks.append((roll, row[name_col], val_num))
    top5 = sorted(marks, key=lambda x: x[2], reverse=True)[:5]
    return pd.DataFrame([
        {'Roll No.': roll, 'Name': name, 'Marks': marks}
        for roll, name, marks in top5
    ])

def show_visualizations(semester_summary, subject_df):
    st.markdown('<h3>📊 Visualizations and Analytics</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    # Pie for pass/fail
    with col1:
        fig = px.pie(
            values=[semester_summary['No. Students Passed'].values[0], semester_summary['No. Students Failed'].values[0]],
            names=['Passed', 'Failed'],
            title='Overall Pass/Fail Distribution',
            color_discrete_sequence=['#4CAF50', '#FF4B4B']
        )
        st.plotly_chart(fig, use_container_width=True)
    # Pie for backlogs
    with col2:
        fig2 = px.pie(
            values=[
                semester_summary['No of Students With 1 Backlog'].values[0],
                semester_summary['No of Students With 2 Backlog'].values[0],
                semester_summary['No of Students With more than 2 Backlog'].values[0]
            ],
            names=['1 Backlog', '2 Backlogs', '3+ Backlogs'],
            title='Backlog Distribution',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig2, use_container_width=True)
    # Bar for pass percentage
    st.markdown("<h4>Pass Percentage by Subject</h4>", unsafe_allow_html=True)
    fig3 = px.bar(subject_df, x='Subject Name', y='Pass Percentage', color='Pass Percentage', text='Pass Percentage')
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig3, use_container_width=True)

uploaded_file = st.file_uploader("Upload the result Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0, header=[6,7])
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df = flatten_headers(df)
        roll_col = next(col for col in df.columns if 'roll no' in col.lower())
        name_col = next(col for col in df.columns if 'student name' in col.lower())
        df['Section'] = df[roll_col].astype(str).apply(lambda x: x[-1] if x and x[-1].isalpha() else 'A')
        subject_blocks = extract_subject_blocks(df)
        subject_blocks = [block for block in subject_blocks if clean_numeric(df[block['external']]).notna().sum() > 0]
        subject_results = []
        failed_dfs = {}
        for block in subject_blocks:
            result, failed = analyze_subject(df, block, roll_col, name_col)
            if result:
                subject_results.append(result)
                if not failed.empty:
                    failed_dfs[block['name']] = failed
        semester_summary = generate_semester_summary(df, subject_blocks, roll_col)
        subject_df = pd.DataFrame(subject_results)
        # Tabs for the dashboard
        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary", "Subject Results", "Failed Students", "Analytics & Top Performers"
        ])
        with tab1:
            st.markdown('<h2>Summary</h2>', unsafe_allow_html=True)
            st.dataframe(semester_summary, use_container_width=True)
        with tab2:
            st.markdown('<h2>Subject-wise Results</h2>', unsafe_allow_html=True)
            st.dataframe(subject_df, use_container_width=True)
        with tab3:
            st.markdown('<h2>Failed Students</h2>', unsafe_allow_html=True)
            if failed_dfs:
                for subj, failed_df in failed_dfs.items():
                    with st.expander(f"{subj} Failed Students ({len(failed_df)})"):
                        st.dataframe(failed_df)
            else:
                st.write("No failed students.")
        with tab4:
            st.markdown('<h2>Academic and Top Performers</h2>', unsafe_allow_html=True)
            show_visualizations(semester_summary, subject_df)
            st.subheader("Top 5 Performers Overall")
            st.dataframe(top_performers_overall(df, subject_blocks, roll_col, name_col), use_container_width=True)
            st.subheader("Top 5 Performers Subject-wise")
            subject_names = [block['name'] for block in subject_blocks]
            selected_subject = st.selectbox("Select Subject", subject_names, key="top5_subject_select")
            block = next(b for b in subject_blocks if b['name'] == selected_subject)
            st.dataframe(top_performers_subject(df, block, roll_col, name_col), use_container_width=True)
        # Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            subject_df.to_excel(writer, sheet_name='Subject Results', index=False)
            for subj, failed_df in failed_dfs.items():
                failed_df.to_excel(writer, sheet_name=sanitize_sheet_name(subj[:20]), index=False)
            semester_summary.to_excel(writer, sheet_name='Semester Summary', index=False)
        st.download_button(
            "Download Full Report",
            output.getvalue(),
            file_name="result_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()
