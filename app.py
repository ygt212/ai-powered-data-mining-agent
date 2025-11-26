import streamlit as st
import pandas as pd
import google.generativeai as genai
from io import StringIO
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori, association_rules
import re
import time
import numpy as np

# --- Constants ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lütfen .streamlit/secrets.toml dosyasını oluşturun ve GOOGLE_API_KEY ekleyin. / Please create .streamlit/secrets.toml and add GOOGLE_API_KEY.")
    st.stop()

MODEL_NAME = "gemini-2.0-flash"

# --- Translations ---
TRANSLATIONS = {
    "tr": {
        "page_title": "Yapay Zeka Destekli Veri Madenciliği Ajanı",
        "header_subtitle": "Google Gemini Altyapısı ile",
        "sidebar_upload": "Veri Yükleme",
        "upload_label": "Veri Seti Yükle (CSV/Excel)",
        "technique_label": "Madencilik Tekniği Seçin",
        "techniques": ["Kümeleme (Clustering)", "Sınıflandırma (Classification)", "Birliktelik Kuralı Madenciliği (Association Rule Mining)"],
        "technique_map": {
            "Kümeleme (Clustering)": "Clustering",
            "Sınıflandırma (Classification)": "Classification",
            "Birliktelik Kuralı Madenciliği (Association Rule Mining)": "Association Rule Mining"
        },
        "bin_count_label": "Sayısal Veri Kategorizasyon Sayısı (Bin Count)",
        "preview_header": "1. Veri Önizleme",
        "preprocessing_header": "2. Yapay Zeka Destekli Ön İşleme",
        "analyze_button": "Veriyi Analiz Et ve İşle",
        "success_processed": "Veri başarıyla işlendi!",
        "tab_original": "Orijinal Veri",
        "tab_cleaned": "AI Tarafından Temizlenmiş Veri",
        "download_csv": "Temizlenmiş CSV İndir",
        "execution_header": "3. {} Yürütme",
        "clustering_running": "K-Means Kümeleme Çalıştırılıyor...",
        "elbow_expander": "İdeal Küme Sayısı Önerisi (Elbow Metodu)",
        "elbow_title": "Elbow Metodu ile İdeal k Değeri",
        "elbow_y_axis": "WCSS (Hata Kareler Toplamı)",
        "cluster_k_slider": "Küme Sayısı (k) Seçin",
        "cluster_no_numeric": "Kümeleme için sayısal veri bulunamadı. Ön işlemeyi kontrol edin.",
        "cluster_result_text": "{} gruba kümelendi.",
        "pca_title": "K-Means Kümeleme (k={}) - PCA Projeksiyonu",
        "cluster_summary_title": "K-Means Kümeleme Analizi (k={}):\n\n",
        "cluster_header": "### Küme {} (Kayıt Sayısı: {})\n",
        "avg_label": "Ortalama",
        "mode_label": "En Sık",
        "classification_running": "Random Forest Sınıflandırma Çalıştırılıyor...",
        "target_select": "Hedef Değişkeni Seçin (Target)",
        "feature_select": "Öznitelik Sütunlarını Seçin (Features)",
        "train_button": "Modeli Eğit",
        "error_no_feature": "Lütfen en az bir öznitelik sütunu seçin.",
        "warning_target_nan": "Hedef değişken '{}' eksik değerler içeriyor. {} satır siliniyor.",
        "info_encoding": "Kategorik değişkenler kodlanıyor...",
        "warning_input_nan": "Girdi verilerinde eksik değerler var. 0 ile dolduruluyor.",
        "error_empty_feature": "Öznitelik seti boş. Model eğitilemez.",
        "accuracy_metric": "Model Doğruluğu (Accuracy)",
        "classification_report_header": "Sınıflandırma Raporu:",
        "confusion_matrix_title": "Karmaşıklık Matrisi (Confusion Matrix)",
        "cm_x_label": "Tahmin Edilen",
        "cm_y_label": "Gerçek Değer",
        "feature_importance_title": "En Önemli 10 Öznitelik",
        "apriori_running": "Apriori Algoritması Çalıştırılıyor...",
        "min_support_slider": "Minimum Destek Eşiği (Support)",
        "run_apriori_button": "Apriori'yi Çalıştır",
        "info_binning": "{} sayısal sütun {} kategoriye ayrılıyor.",
        "info_onehot": "One-Hot Encoding uygulanıyor...",
        "info_processing_items": "{} öğe (sütun) işleniyor...",
        "warning_no_frequent": "Destek değeri >= {} olan sık öğe seti bulunamadı. Eşiği düşürmeyi deneyin.",
        "warning_no_rules": "Sık öğe setleri bulundu ancak lift eşiğini geçen kural yok.",
        "success_rules_found": "{} kural bulundu.",
        "rules_scatter_title": "Birliktelik Kuralları (Support vs Confidence)",
        "top_rules_header": "En Güçlü Birliktelik Kuralları (Lift'e göre sıralı):",
        "interpretation_header": "Yapay Zeka İş Yorumu",
        "interpret_button": "Sonuçları Yorumla",
        "download_results_button": "Sonuçları ve Yorumu İndir",
        "ai_prompt_lang": "TURKISH",
        "spinner_cleaning": "Yapay Zeka temizleme kodunu oluşturuyor...",
        "spinner_running_code": "Temizleme kodu çalıştırılıyor...",
        "spinner_interpreting": "Yapay Zeka sonuçları yorumluyor...",
        "error_api_key": "API Anahtarı Yapılandırma Hatası: {}",
        "error_gemini": "Gemini API Hatası: {}",
        "error_code_exec": "Kod yürütme hatası: {}",
        "error_ai_clean_func": "Yapay Zeka 'clean_data' fonksiyonunu oluşturamadı.",
        "error_file_upload": "Dosya yükleme hatası: {}",
        "error_clustering": "Kümeleme Hatası: {}",
        "error_classification": "Sınıflandırma Hatası: {}",
        "error_apriori": "Birliktelik Kuralı Hatası: {}",
        "rate_limit_warning": "Hız sınırı aşıldı (429). {} saniye içinde tekrar deneniyor... (Deneme {}/{})",
        "welcome_title": "Nasıl Kullanılır?",
        "welcome_message": "Lütfen başlamak için sol menüden bir veri seti yükleyin.",
        "step_1": "Veri Yükle: Sol menüden dosyanızı seçin.",
        "step_2": "Teknik Seç: Kümeleme, Sınıflandırma veya Birliktelik.",
        "step_3": "Analiz Et: Yapay zeka verinizi işlesin.",
        "step_4": "Yorumla: İş dünyası içgörülerini okuyun.",
        "uploader_text": "Dosyayı buraya sürükleyip bırakın",
        "uploader_limit_text": "Dosya limiti: 200MB • CSV, XLSX",
        "uploader_button_text": "Dosyalara Gözat"
    },
    "en": {
        "page_title": "AI-Powered Data Mining Agent",
        "header_subtitle": "Powered by Google Gemini",
        "sidebar_upload": "Data Upload",
        "upload_label": "Upload Dataset (CSV/Excel)",
        "technique_label": "Select Mining Technique",
        "techniques": ["Clustering", "Classification", "Association Rule Mining"],
        "technique_map": {
            "Clustering": "Clustering",
            "Classification": "Classification",
            "Association Rule Mining": "Association Rule Mining"
        },
        "bin_count_label": "Numeric Data Categorization Count (Bin Count)",
        "preview_header": "1. Data Preview",
        "preprocessing_header": "2. AI-Powered Preprocessing",
        "analyze_button": "Analyze & Process Data",
        "success_processed": "Data successfully processed!",
        "tab_original": "Original Data",
        "tab_cleaned": "AI Cleaned Data",
        "download_csv": "Download Cleaned CSV",
        "execution_header": "3. {} Execution",
        "clustering_running": "Running K-Means Clustering...",
        "elbow_expander": "Optimal Cluster Count Suggestion (Elbow Method)",
        "elbow_title": "Optimal k Value via Elbow Method",
        "elbow_y_axis": "WCSS (Within-Cluster Sum of Squares)",
        "cluster_k_slider": "Select Number of Clusters (k)",
        "cluster_no_numeric": "No numeric data found for clustering. Check preprocessing.",
        "cluster_result_text": "Clustered into {} groups.",
        "pca_title": "K-Means Clustering (k={}) - PCA Projection",
        "cluster_summary_title": "K-Means Clustering Analysis (k={}):\n\n",
        "cluster_header": "### Cluster {} (Count: {})\n",
        "avg_label": "Average",
        "mode_label": "Most Frequent",
        "classification_running": "Running Random Forest Classification...",
        "target_select": "Select Target Variable",
        "feature_select": "Select Feature Columns",
        "train_button": "Train Model",
        "error_no_feature": "Please select at least one feature column.",
        "warning_target_nan": "Target variable '{}' contains missing values. {} rows dropped.",
        "info_encoding": "Encoding categorical variables...",
        "warning_input_nan": "Input data contains missing values. Filling with 0.",
        "error_empty_feature": "Feature set is empty. Cannot train model.",
        "accuracy_metric": "Model Accuracy",
        "classification_report_header": "Classification Report:",
        "confusion_matrix_title": "Confusion Matrix",
        "cm_x_label": "Predicted",
        "cm_y_label": "Actual",
        "feature_importance_title": "Top 10 Important Features",
        "apriori_running": "Running Apriori Algorithm...",
        "min_support_slider": "Minimum Support Threshold",
        "run_apriori_button": "Run Apriori",
        "info_binning": "{} numeric columns are being split into {} categories.",
        "info_onehot": "Applying One-Hot Encoding...",
        "info_processing_items": "Processing {} items (columns)...",
        "warning_no_frequent": "No frequent itemsets found with support >= {}. Try lowering the threshold.",
        "warning_no_rules": "Frequent itemsets found, but no rules passed the lift threshold.",
        "success_rules_found": "{} rules found.",
        "rules_scatter_title": "Association Rules (Support vs Confidence)",
        "top_rules_header": "Strongest Association Rules (Sorted by Lift):",
        "interpretation_header": "AI Business Interpretation",
        "interpret_button": "Interpret Results",
        "download_results_button": "Download Results & Interpretation",
        "ai_prompt_lang": "ENGLISH",
        "spinner_cleaning": "AI is generating cleaning code...",
        "spinner_running_code": "Running cleaning code...",
        "spinner_interpreting": "AI is interpreting results...",
        "error_api_key": "API Key Configuration Error: {}",
        "error_gemini": "Gemini API Error: {}",
        "error_code_exec": "Code execution error: {}",
        "error_ai_clean_func": "AI failed to generate 'clean_data' function.",
        "error_file_upload": "File upload error: {}",
        "error_clustering": "Clustering Error: {}",
        "error_classification": "Classification Error: {}",
        "error_apriori": "Association Rule Error: {}",
        "rate_limit_warning": "Rate limit exceeded (429). Retrying in {} seconds... (Attempt {}/{})",
        "welcome_title": "How to Use",
        "welcome_message": "Please upload a dataset from the sidebar to get started.",
        "step_1": "Upload Data: Select file from sidebar.",
        "step_2": "Select Technique: Clustering, Classification, or Association.",
        "step_3": "Analyze: Let AI process your data.",
        "step_4": "Interpret: Read business insights.",
        "uploader_text": "Drag and drop file here",
        "uploader_limit_text": "Limit 200MB per file • CSV, XLSX",
        "uploader_button_text": "Browse files"
    }
}

# --- Configuration ---
st.set_page_config(page_title="Yapay Zeka Destekli Veri Madenciliği Ajanı / AI-Powered Data Mining Agent", layout="wide")

# --- Helper Functions ---

def configure_gemini():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        return True
    except Exception as e:
        st.error(f"API Key Error: {e}")
        return False

@st.cache_data(show_spinner=False)
def get_gemini_response(prompt):
    if not configure_gemini():
        return None
    
    model = genai.GenerativeModel(MODEL_NAME)
    
    max_retries = 5
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Resource exhausted" in error_str:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    st.warning(f"Rate limit (429). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            
            st.error(f"Gemini API Error: {e}")
            return None
    return None

def extract_code_from_text(text):
    """Extracts Python code from Gemini's response."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.replace("```python", "").replace("```", "")

@st.cache_data(show_spinner=False)
def preprocess_with_gemini(df, technique, lang_code):
    t = TRANSLATIONS[lang_code]
    
    # Capture df.info()
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Capture head
    head_str = df.head(10).to_string()
    
    # System prompt can remain largely English as it's for the model's internal logic
    system_prompt = f"""
    You are an Expert Python Data Engineer. 
    The user wants to perform '{technique}' on a dataset.
    
    Dataset Info:
    {info_str}
    
    First 10 Rows:
    {head_str}
    
    Task: Write a Python function named `clean_data(df)` that takes the pandas DataFrame `df` as input and returns a cleaned DataFrame.
    
    Requirements for `clean_data(df)`:
    1. Handle missing values appropriately for {technique} (impute or drop).
    2. Encode categorical variables IF necessary for general cleaning, but prefer keeping them readable if the specific technique (like Association Rules) handles them later. 
       - For Clustering/Classification: You might want to LabelEncode or OneHotEncode if the column is clearly categorical.
       - For Association Rules: Keep categorical values as they are; we will handle discretization later.
    3. Normalize numerical values if critical for {technique} (e.g. K-Means).
    4. Handle any obvious data type issues (e.g. object columns that should be numeric).
    5. The function MUST return the modified DataFrame.
    6. Include all necessary imports inside the function (e.g. `import pandas as pd`, `from sklearn.preprocessing import LabelEncoder`).
    
    RETURN ONLY THE PYTHON CODE. No markdown explanations.
    """
    
    with st.spinner(t["spinner_cleaning"]):
        response_text = get_gemini_response(system_prompt)
        
    if response_text:
        code_str = extract_code_from_text(response_text)
        
        try:
            local_scope = {}
            exec(code_str, globals(), local_scope)
            
            if 'clean_data' in local_scope:
                with st.spinner(t["spinner_running_code"]):
                    cleaned_df = local_scope['clean_data'](df.copy())
                return cleaned_df
            else:
                st.error(t["error_ai_clean_func"])
                return None
        except Exception as e:
            st.error(t["error_code_exec"].format(e))
            return None
    return None

def interpret_results_with_gemini(results_summary, lang_code):
    t = TRANSLATIONS[lang_code]
    prompt = f"""
    You are a Business Intelligence Analyst. 
    Interpret the following data mining results and provide a business-oriented explanation in {t['ai_prompt_lang']}.
    Focus on actionable insights.
    
    IMPORTANT: Do NOT use markdown bold syntax (double asterisks like **text**). Provide the output in clean plain text, using only bullet points or headers if necessary, but no bold styling.
    
    Results:
    {results_summary}
    """
    with st.spinner(t["spinner_interpreting"]):
        return get_gemini_response(prompt)

# --- Custom CSS ---
def local_css():
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Main Background - Dark Theme */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #262730;
            border-right: 1px solid #41444C;
        }
        
        /* Badge Styling */
        .badge {
            background-color: #2f3061; /* Indigo-600 */
            color: white !important;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 4px rgba(79, 70, 229, 0.4);
        }
        
        /* Button Styling Overrides */
        button[kind="primary"] {
            background-color: #2f3061;
            border-color: #2f3061;
            color: white;
        }
        button[kind="primary"]:hover {
            background-color: #2A2A54;
            border-color: #2A2A54;
            color: white;
        }
        
        /* Text Color Overrides for Dark Mode Compatibility */
        h1, h2, h3, h4, h5, h6, p, li, span, div, label {
            color: #FAFAFA !important;
        }
        
        /* Metric Value Color */
        [data-testid="stMetricValue"] {
            color: #FAFAFA !important;
        }
        
        /* Dataframe/Table Text */
        [data-testid="stDataFrame"] {
            color: #FAFAFA !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Main App ---

def main():
    local_css()
    
    # --- Language Selector (Top Right) ---
    col1, col2 = st.columns([10, 2])
    
    with col2:
        lang_selection = st.selectbox(
            "Change Language / Dil Değiştir", 
            ["Türkçe", "English"], 
            label_visibility="visible"
        )
    
    lang_code = "tr" if lang_selection == "Türkçe" else "en"
    t = TRANSLATIONS[lang_code]
    
    # --- Hack: Translate/Style File Uploader Internal Text via CSS ---
    # Applied to BOTH languages for consistent design
    st.markdown(f"""
        <style>
        /* Hide original text */
        [data-testid="stFileUploaderDropzone"] div div span {{display: none;}}
        [data-testid="stFileUploaderDropzone"] div div small {{display: none;}}
        
        /* Insert Custom Text */
        [data-testid="stFileUploaderDropzone"] div div::before {{
            content: "{t['uploader_text']}";
            display: block;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        [data-testid="stFileUploaderDropzone"] div div::after {{
            content: "{t['uploader_limit_text']}";
            display: block;
            font-size: 0.8rem;
            opacity: 0.7;
        }}
        
        /* Button Text Hack */
        [data-testid="stFileUploaderDropzone"] button {{
            color: transparent !important;
            position: relative;
        }}
        [data-testid="stFileUploaderDropzone"] button::after {{
            content: "{t['uploader_button_text']}";
            color: #FAFAFA; /* White text */
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-weight: 400;
            font-size: 14px;
            line-height: 1;
            white-space: nowrap;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    with col1:
        st.title(t["page_title"])
        st.markdown(f'<span class="badge">{t["header_subtitle"]}</span>', unsafe_allow_html=True)
    
    st.divider()

    # --- Sidebar ---
    st.sidebar.header(t["sidebar_upload"])
    
    uploaded_file = st.sidebar.file_uploader(t["upload_label"], type=["csv", "xlsx"])
    
    technique = st.sidebar.selectbox(
        t["technique_label"], 
        t["techniques"]
    )
    
    # Map selection back to internal keys
    selected_technique_key = t["technique_map"][technique]

    # --- Auto-Reset Logic ---
    if "last_technique" not in st.session_state:
        st.session_state.last_technique = selected_technique_key
    
    if st.session_state.last_technique != selected_technique_key:
        st.session_state.cleaned_df = None
        st.session_state.results_summary = None
        st.session_state.interpretation = None
        # Clear Classification State
        st.session_state.class_acc = None
        st.session_state.class_report = None
        st.session_state.class_fig = None
        st.session_state.class_cm_fig = None
        # Clear Apriori State
        st.session_state.apriori_rules = None
        st.session_state.apriori_fig = None
        
        st.session_state.last_technique = selected_technique_key
        st.rerun()

    if uploaded_file:
        # Load Data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(t["error_file_upload"].format(e))
            return

        with st.container(border=True):
            st.subheader(t["preview_header"])
            st.dataframe(df.head(), use_container_width=True)

        # --- Phase 1: AI Preprocessing ---
        with st.container(border=True):
            st.subheader(t["preprocessing_header"])
            
            if "cleaned_df" not in st.session_state:
                st.session_state.cleaned_df = None
            
            if "results_summary" not in st.session_state:
                st.session_state.results_summary = None
                
            if "interpretation" not in st.session_state:
                st.session_state.interpretation = None

            if st.button(t["analyze_button"], type="primary"):
                cleaned_df = preprocess_with_gemini(df, selected_technique_key, lang_code)
                if cleaned_df is not None:
                    st.session_state.cleaned_df = cleaned_df
                    # Clear downstream results if data changes
                    st.session_state.results_summary = None
                    st.session_state.interpretation = None
                    st.session_state.class_acc = None
                    st.session_state.class_report = None
                    st.session_state.class_fig = None
                    st.session_state.class_cm_fig = None
                    st.session_state.apriori_rules = None
                    st.session_state.apriori_fig = None
                    st.success(t["success_processed"])

            if st.session_state.cleaned_df is not None:
                cleaned_df = st.session_state.cleaned_df
                
                tab1, tab2 = st.tabs([t["tab_original"], t["tab_cleaned"]])
                with tab1:
                    st.dataframe(df, use_container_width=True)
                with tab2:
                    st.dataframe(cleaned_df, use_container_width=True)
                    st.download_button(
                        t["download_csv"],
                        cleaned_df.to_csv(index=False),
                        "temizlenmis_veri.csv",
                        "text/csv"
                    )

        # --- Phase 2 & 3: Execution & Interpretation ---
        # Only show if data is cleaned and ready
        if st.session_state.cleaned_df is not None:
            cleaned_df = st.session_state.cleaned_df
            
            st.subheader(t["execution_header"].format(technique))
            
            if selected_technique_key == "Clustering":
                with st.container(border=True):
                    st.write(t["clustering_running"])
                    
                    # Elbow Method (WSS)
                    try:
                        numeric_df_elbow = cleaned_df.select_dtypes(include=['number'])
                        if not numeric_df_elbow.empty:
                            if numeric_df_elbow.isna().any().any():
                                numeric_df_elbow = numeric_df_elbow.fillna(numeric_df_elbow.mean())
                            
                            with st.expander(t["elbow_expander"], expanded=False):
                                wcss = []
                                k_range = range(1, 11)
                                for i in k_range:
                                    kmeans_elbow = KMeans(n_clusters=i, random_state=42)
                                    kmeans_elbow.fit(numeric_df_elbow)
                                    wcss.append(kmeans_elbow.inertia_)
                                
                                fig_elbow = px.line(
                                    x=list(k_range), 
                                    y=wcss, 
                                    markers=True,
                                    labels={'x': 'k', 'y': t["elbow_y_axis"]},
                                    title=t["elbow_title"]
                                )
                                st.plotly_chart(fig_elbow, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Elbow Error: {e}")

                    k = st.slider(t["cluster_k_slider"], 2, 10, 3)
                    
                    try:
                        # Select numeric columns from CLEANED data for training
                        numeric_df = cleaned_df.select_dtypes(include=['number'])
                        
                        if numeric_df.empty:
                            st.error(t["cluster_no_numeric"])
                        else:
                            # Handle NaNs if any remain
                            if numeric_df.isna().any().any():
                                numeric_df = numeric_df.fillna(numeric_df.mean())
                                
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            clusters = kmeans.fit_predict(numeric_df)
                            
                            # 1. Map Clusters to ORIGINAL Data for interpretation
                            df_with_clusters = df.copy()
                            df_with_clusters['Küme'] = clusters
                            
                            st.metric("Clusters", f"{k}")
                            st.dataframe(df_with_clusters.head(), use_container_width=True)
                            
                            # 2. Visualization (PCA on numeric/cleaned data)
                            if numeric_df.shape[1] >= 2:
                                pca = PCA(n_components=2)
                                components = pca.fit_transform(numeric_df)
                                
                                pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                                pca_df['Küme'] = clusters.astype(str)
                                
                                # Add hover columns from ORIGINAL data (more readable)
                                hover_cols = df_with_clusters.columns[:5].tolist()
                                if 'Küme' in hover_cols: hover_cols.remove('Küme')
                                    
                                for col in hover_cols:
                                    # Ensure lengths match (they should)
                                    pca_df[col] = df_with_clusters[col].values
                                
                                # Interactive Plotly Chart
                                fig = px.scatter(
                                    pca_df, 
                                    x='PC1', 
                                    y='PC2', 
                                    color='Küme',
                                    title=t["pca_title"].format(k),
                                    hover_data=hover_cols 
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # 3. Generate Human-Readable Summary
                            summary = t["cluster_summary_title"].format(k)
                            
                            for i in range(k):
                                cluster_sub = df_with_clusters[df_with_clusters['Küme'] == i]
                                size = len(cluster_sub)
                                summary += t["cluster_header"].format(i, size)
                                
                                # Summarize columns
                                for col in df.columns:
                                    if col == 'Küme': continue
                                    
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        mean_val = cluster_sub[col].mean()
                                        summary += f"- {col} ({t['avg_label']}): {mean_val:.2f}\n"
                                    else:
                                        # Categorical
                                        if not cluster_sub[col].empty:
                                            top_val = cluster_sub[col].mode()
                                            if not top_val.empty:
                                                val = top_val.iloc[0]
                                                summary += f"- {col} ({t['mode_label']}): {val}\n"
                                summary += "\n"
                            
                            # Only update if changed to avoid loop, but here we just set it
                            st.session_state.results_summary = summary

                    except Exception as e:
                        st.error(t["error_clustering"].format(e))

            elif selected_technique_key == "Classification":
                with st.container(border=True):
                    st.write(t["classification_running"])
                    
                    # Target Selection
                    target_col = st.selectbox(t["target_select"], cleaned_df.columns, index=len(cleaned_df.columns)-1)
                    
                    # Feature Selection
                    all_cols = cleaned_df.columns.tolist()
                    if target_col in all_cols:
                        all_cols.remove(target_col)
                    feature_cols = st.multiselect(t["feature_select"], all_cols, default=all_cols)
                    
                    if st.button(t["train_button"], type="primary"):
                        if not feature_cols:
                            st.error(t["error_no_feature"])
                        else:
                            try:
                                X = cleaned_df[feature_cols]
                                y = cleaned_df[target_col]
                                
                                # High Cardinality Guardrail for Classification
                                dropped_cols = []
                                for col in X.columns:
                                    if not pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 50:
                                        dropped_cols.append(col)
                                
                                if dropped_cols:
                                    X = X.drop(columns=dropped_cols)
                                    st.warning(f"High cardinality features dropped (>50 unique values): {', '.join(dropped_cols)}")
                                
                                # Drop rows where target is NaN
                                if y.isna().any():
                                    st.warning(t["warning_target_nan"].format(target_col, y.isna().sum()))
                                    valid_indices = y.dropna().index
                                    X = X.loc[valid_indices]
                                    y = y.loc[valid_indices]
                                
                                # Force Label Encoding for Target Variable (Fix for Unknown label type: continuous)
                                le = LabelEncoder()
                                y = y.astype(str)
                                y = le.fit_transform(y)

                                # Handle Features
                                X_numeric = X.select_dtypes(include=['number'])
                                
                                # Fallback: If no numeric columns or mixed, encode categorical
                                if X_numeric.shape[1] != X.shape[1]:
                                    st.info(t["info_encoding"])
                                    X = pd.get_dummies(X)
                                
                                # Fill NaNs in X
                                if X.isna().any().any():
                                    st.warning(t["warning_input_nan"])
                                    X = X.fillna(0)
                                    
                                if X.empty:
                                    st.error(t["error_empty_feature"])
                                else:
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                    
                                    clf = RandomForestClassifier(random_state=42)
                                    clf.fit(X_train, y_train)
                                    y_pred = clf.predict(X_test)
                                    
                                    acc = accuracy_score(y_test, y_pred)
                                    report = classification_report(y_test, y_pred, output_dict=True)
                                    report_text = classification_report(y_test, y_pred)
                                    
                                    # Confusion Matrix
                                    cm = confusion_matrix(y_test, y_pred)
                                    fig_cm = px.imshow(
                                        cm, 
                                        text_auto=True,
                                        labels=dict(x=t["cm_x_label"], y=t["cm_y_label"]),
                                        x=[str(c) for c in le.classes_],
                                        y=[str(c) for c in le.classes_],
                                        title=t["confusion_matrix_title"]
                                    )
                                    
                                    # Feature Importance Plot
                                    if hasattr(clf, 'feature_importances_'):
                                        importances = pd.DataFrame({
                                            'Feature': X.columns,
                                            'Importance': clf.feature_importances_
                                        }).sort_values(by='Importance', ascending=False).head(10)
                                        
                                        fig = px.bar(
                                            importances, 
                                            x='Importance', 
                                            y='Feature', 
                                            orientation='h',
                                            title=t["feature_importance_title"]
                                        )
                                    else:
                                        fig = None
                                    
                                    # Save to session state
                                    st.session_state.class_acc = acc
                                    st.session_state.class_report = report_text
                                    st.session_state.class_fig = fig
                                    st.session_state.class_cm_fig = fig_cm
                                    
                                    results_summary = f"Random Forest Classification on target '{target_col}'.\n"
                                    results_summary += f"Accuracy: {acc:.2f}\n"
                                    results_summary += f"Classification Report:\n{report_text}"
                                    st.session_state.results_summary = results_summary
                                    # Clear interpretation if model retrained
                                    st.session_state.interpretation = None
                                
                            except Exception as e:
                                st.error(t["error_classification"].format(e))
                
                # Render results from session state
                if "class_acc" in st.session_state and st.session_state.class_acc is not None:
                    with st.container(border=True):
                        st.metric(t["accuracy_metric"], f"{st.session_state.class_acc:.2f}")
                        st.text(t["classification_report_header"])
                        st.text(st.session_state.class_report)
                        
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            if "class_cm_fig" in st.session_state and st.session_state.class_cm_fig:
                                st.plotly_chart(st.session_state.class_cm_fig, use_container_width=True)
                        with col_chart2:
                            if st.session_state.class_fig:
                                st.plotly_chart(st.session_state.class_fig, use_container_width=True)

            elif selected_technique_key == "Association Rule Mining":
                with st.container(border=True):
                    st.write(t["apriori_running"])
                    
                    bin_count = st.slider(t["bin_count_label"], 2, 5, 3)
                    min_support = st.slider(t["min_support_slider"], 0.001, 0.5, 0.05, 0.001)
                    
                    if st.button(t["run_apriori_button"], type="primary"):
                        try:
                            # 1. Discretize Numeric Columns
                            # Create a copy to avoid mutating session state directly if we re-run
                            mining_df = cleaned_df.copy()
                            
                            numeric_cols = mining_df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                st.info(t["info_binning"].format(len(numeric_cols), bin_count))
                                for col in numeric_cols:
                                    # Use qcut for quantile-based discretization, fallback to cut if unique values are few
                                    try:
                                        mining_df[col] = pd.qcut(mining_df[col], q=bin_count, labels=False, duplicates='drop')
                                        mining_df[col] = mining_df[col].astype(str)
                                    except ValueError:
                                        # If edges are not unique (e.g. many 0s), use cut or just convert to string
                                        mining_df[col] = mining_df[col].astype(str)
                            
                            # High Cardinality Guardrail for Association Rules
                            dropped_cols_apriori = []
                            for col in mining_df.columns:
                                if mining_df[col].nunique() > 50:
                                    dropped_cols_apriori.append(col)
                            
                            if dropped_cols_apriori:
                                mining_df = mining_df.drop(columns=dropped_cols_apriori)
                                st.warning(f"High cardinality columns dropped (>50 unique values): {', '.join(dropped_cols_apriori)}")

                            # 2. One-Hot Encoding
                            st.write(t["info_onehot"])
                            bool_df = pd.get_dummies(mining_df)
                            bool_df = bool_df.astype(bool)
                            
                            st.write(t["info_processing_items"].format(bool_df.shape[1]))
                            
                            frequent_itemsets = apriori(bool_df, min_support=min_support, use_colnames=True)
                            
                            if frequent_itemsets.empty:
                                st.warning(t["warning_no_frequent"].format(min_support))
                                st.session_state.results_summary = "No frequent itemsets found."
                                st.session_state.apriori_rules = None
                                st.session_state.apriori_fig = None
                            else:
                                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                                
                                if rules.empty:
                                    st.warning(t["warning_no_rules"])
                                    st.session_state.results_summary = "Frequent itemsets found, but no association rules."
                                    st.session_state.apriori_rules = None
                                    st.session_state.apriori_fig = None
                                else:
                                    st.success(t["success_rules_found"].format(len(rules)))
                                    
                                    # Interactive Scatter Plot for Rules
                                    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                                    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                                    
                                    fig = px.scatter(
                                        rules, 
                                        x="support", 
                                        y="confidence", 
                                        size="lift", 
                                        color="lift",
                                        hover_data=["antecedents_str", "consequents_str"],
                                        title=t["rules_scatter_title"]
                                    )
                                    
                                    # Save to session state
                                    st.session_state.apriori_rules = rules
                                    st.session_state.apriori_fig = fig
                                    
                                    results_summary = "Association Rule Mining Results:\n"
                                    results_summary += rules.sort_values(by="lift", ascending=False).head(5).to_string()
                                    st.session_state.results_summary = results_summary
                                    # Clear interpretation if new rules found
                                    st.session_state.interpretation = None
                                
                        except Exception as e:
                            st.error(t["error_apriori"].format(e))
                
                # Render results from session state
                if "apriori_rules" in st.session_state and st.session_state.apriori_rules is not None:
                    with st.container(border=True):
                        if st.session_state.apriori_fig:
                            st.plotly_chart(st.session_state.apriori_fig, use_container_width=True)
                        
                        st.write(t["top_rules_header"])
                        st.dataframe(st.session_state.apriori_rules.sort_values(by="lift", ascending=False).head(10), use_container_width=True)

            # --- Phase 3: Interpretation ---
            if st.session_state.results_summary:
                with st.container(border=True):
                    st.subheader(t["interpretation_header"].format(t["ai_prompt_lang"]))
                    
                    if st.button(t["interpret_button"], type="primary"):
                        interpretation = interpret_results_with_gemini(st.session_state.results_summary, lang_code)
                        if interpretation:
                            st.session_state.interpretation = interpretation
                    
                    if st.session_state.interpretation:
                        st.markdown(st.session_state.interpretation)
                        
                        st.download_button(
                            t["download_results_button"],
                            f"Results:\n{st.session_state.results_summary}\n\nInterpretation:\n{st.session_state.interpretation}",
                            "analiz_sonucu.txt",
                            "text/plain"
                        )
    else:
        # --- Empty State (Onboarding) ---
        with st.container(border=True):
            st.subheader(t["welcome_title"])
            st.write(t["welcome_message"])
            
            st.markdown(f"""
            1. **{t['step_1']}**
            2. **{t['step_2']}**
            3. **{t['step_3']}**
            4. **{t['step_4']}**
            """)

if __name__ == "__main__":
    main()
