import streamlit as st
import numpy as np
from scipy import stats
import math
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f


#=======================================================================================================
#-----------------------------------Konfigurasi Situs
#===============================================================================================================

st.set_page_config(page_title="Uji Statistika Parametrik", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fredoka+One&display=swap" rel="stylesheet">

<style>

    /* ============================
       MAIN HEADER (Judul)
       ============================ */
    .neon-title {
        font-family: 'Fredoka One', cursive;
        font-size: 2.5rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-align: center;
        padding: 25px;
        color: #0B3D0B !important;  
        border-radius: 15px;
        margin-bottom: 35px;

        /* Gradient hijau → kuning */
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 40%, #f9f586 100%);

        /* Border & shadow */
        border: 3px solid #7cb342;
        box-shadow: 0 4px 20px rgba(124, 187, 55, 0.35);
    }

    /* ============================
       SUB HEADER
       ============================ */
    .sub-header {
        font-size: 1.9rem;
        font-weight: bold;
        color: #2e7d32;
        margin-top: 20px;
        margin-bottom: 15px;
        font-family: 'Lora', serif;
    }

    /* ============================
       INFO BOX
       ============================ */
    .info-box {
        background: linear-gradient(135deg, #d4fc79 0%, #f9f586 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #8bc34a;
        margin: 15px 0;
    }

    .result-box {
        background: linear-gradient(135deg, #96e6a1 0%, #f9f586 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #7cb342;
        margin: 15px 0;
    }

    /* ============================
       SIDEBAR
       ============================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #d4fc79 0%, #f9f586 100%);
        padding: 1.5rem 1rem;
    }

    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        padding: 14px 18px;
        margin: 5px 0;
        border-radius: 12px;
        border: 2px solid #aed581;
        font-weight: 600;
        font-size: 0.95rem;
        background: linear-gradient(135deg, rgba(212, 252, 121, 0.8) 0%, rgba(249, 245, 134, 0.8) 100%);
        color: #33691e;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #ccff90 0%, #f4ff81 100%);
        color: #1b5e20;
        transform: translateX(5px);
        border-color: #9ccc65;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* ============================
       MAIN BUTTONS
       ============================ */
    .stButton>button {
        background: linear-gradient(135deg, #7cb342 0%, #fdd835 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 25px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #558b2f 0%, #fbc02d 100%);
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(85,139,47,0.25);
    }

    /* ============================
       TABS
       ============================ */
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #d4fc79 0%, #f9f586 100%);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #33691e;
        border: 2px solid #aed581;
        font-family: 'Lora', serif;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #ccff90 0%, #ffeb3b 100%);
        color: #1b5e20;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7cb342 0%, #fdd835 100%) !important;
        color: white !important;
        border-color: #7cb342 !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: linear-gradient(135deg, #f1f8e9 0%, #fffde7 100%);
        padding: 20px;
        border-radius: 0 0 8px 8px;
        border: 2px solid #aed581;
        border-top: none;
    }

</style>
""", unsafe_allow_html=True)


# ====================================================================================
#                                  NAVIGASI
# ====================================================================================

st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih modul:", ["Uji Proporsi", "Uji Rata-rata Satu Sampel", "Uji Rata-rata 2 Sampel Independen – Varians Diketahui",
                                        "Uji Kesamaan Varians", "Uji Rata-rata 2 Sampel Independen – Varians Sama",
                                        "Uji Rata-rata 2 Sampel Independen – Varians Tidak Sama",
                                        "Uji Rata-rata 2 Sampel Dependen"])

# ========================================================================================================================================================
#                                  "UJI PROPORSI'
# ==========================================================================================================================================================

if menu == "Uji Proporsi":
    st.title("Uji Proporsi (1 Sampel & 2 Sampel)")

#-----------------------------FUNGSI BANTU------------------------------------------------------------------


    def z_critical(alpha, alt):
        if alt == "dua sisi":
            z = stats.norm.ppf(1 - alpha/2)
            return -z, z
        elif alt == "kiri":
            return -np.inf, stats.norm.ppf(alpha)
        else:  # kanan
            return stats.norm.ppf(1 - alpha), np.inf

    def p_value(z, alt):
        if alt == "dua sisi":
            return 2 * (1 - stats.norm.cdf(abs(z)))
        elif alt == "kiri":
            return stats.norm.cdf(z)
        else:  # kanan
            return 1 - stats.norm.cdf(z)

    tab1, tab2 = st.tabs(["SATU SAMPEL", "DUA SAMPEL"])

# ==============================================================================================================================================================================
#                           UJI PROPORSI SATU SAMPEL
# ==========================================================================================================================================================

    with tab1:

        st.header("Uji Proporsi Satu Sampel")

        sub1, sub2, sub3, sub4 = st.tabs(
            ["Konsep", "Rumus", "Contoh Perhitungan", "Kalkulator"]
        )

        # --------------------- KONSEP ---------------------
        with sub1:
            st.subheader("Konsep Dasar")
            st.write("""
            Uji proporsi **satu sampel** digunakan untuk menguji apakah proporsi populasi
            sama dengan nilai hipotesis p₀.
            
            - Data berasal dari satu sampel
            - Data berupa kategori (sukses/gagal)
            """)

            st.markdown("### Hipotesis Dua Arah")
            st.latex(r"H_0: p = p_0")
            st.latex(r"H_1: p \ne p_0")

            st.markdown("### Hipotesis Satu Arah (Kanan)")
            st.latex(r"H_0: p \le p_0")
            st.latex(r"H_1: p > p_0")

            st.markdown("### Hipotesis Satu Arah (Kiri)")
            st.latex(r"H_0: p \ge p_0")
            st.latex(r"H_1: p < p_0")

        # --------------------- RUMUS ---------------------
        with sub2:
            st.subheader("Rumus Uji Proporsi Satu Sampel")

            st.latex(r"\hat{p} = \frac{X}{n}")
            st.latex(r"Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}")

            st.markdown("""
            **Keterangan Rumus:**
            - **X** : jumlah sukses dalam sampel  
            - **n** : ukuran sampel  
            - **p̂** : proporsi sampel  
            - **p₀** : proporsi hipotesis  
            - **Z** : statistik uji  
            """)

        # --------------------- CONTOH ---------------------
        with sub3:
            st.subheader("Contoh Perhitungan Satu Sampel")

            st.write("""
            Sebuah perusahaan ingin mengetahui apakah proporsi pelanggan yang puas 
            berbeda dari 50%.  
            Dari 50 pelanggan yang disurvei, 30 menyatakan puas. 
            Dengan taraf signifikansi 5% pakah proporsi pelanggan yang puas berbeda dari 50%?
            
            - X = 30  
            - n = 50  
            - p₀ = 0.50  
            - α = 0.05  
            """)

            # Perhitungan contoh
            X = 30
            n = 50
            p0 = 0.50
            p_hat = X/n
            z = (p_hat - p0) / math.sqrt(p0*(1-p0)/n)
            pval = 2 * (1 - stats.norm.cdf(abs(z)))

            st.markdown("### Langkah 1: Hitung Proporsi Sampel")
            st.latex(r"\hat{p} = \frac{30}{50} = 0.60")

            st.markdown("### Langkah 2: Hitung Statistik Z")
            st.latex(
                fr"""
                Z =
                \frac{{0.60 - 0.50}}
                {{\sqrt{{0.50(1-0.50)/50}}}}
                = {z:.4f}
                """
            )

            st.markdown("### Langkah 3: Hitung p-value (Uji Dua Arah)")
            st.write("""Nilai **p-value** dihitung dari *luas daerah ekor distribusi normal standar*
            setelah nilai Z yang dihitung. Karena uji dua arah, maka daerahnya menjadi dua sisi.""")
            st.latex(
                fr"""
                p\text{{-value}}
                = 2(1 - \Phi(|{z:.4f}|))
                = {pval:.4f}
                """
            )

            st.markdown("### Langkah 4: Nilai Kritis (α = 0.05)")
            st.write("""
            Nilai kritis diperoleh dari **quantile distribusi normal standar**.
            Karena ini uji dua arah dengan α = 0.05:
            
            - α/2 = 0.025 pada masing-masing ekor  
            - Maka nilai kritis adalah Z pada kuantil 0.975(karena 1 − 0.025 = 0.975)
            """)
            
            st.latex(r"Z_{\text{krit}} = \pm 1.96")

            st.markdown("### Langkah 5: Keputusan Akhir")

            if abs(z) > 1.96:
                st.markdown(fr"""
                **Tolak H₀**, karena  
                \[
                |Z| = {abs(z):.4f} > 1.96
                \]  
                Terdapat bukti signifikan bahwa proporsi pelanggan **berbeda dari 50\%**.
                """)
            else:
                st.markdown(fr"""
                **Gagal Menolak H₀**, karena  
                \[
                |Z| = {abs(z):.4f} \le 1.96
                \]  
                Tidak terdapat bukti yang cukup bahwa proporsi pelanggan **berbeda dari 50\%**.
                """)


        # --------------------- KALKULATOR ---------------------
        with sub4:

            st.subheader("Kalkulator Proporsi Satu Sampel")

            col1, col2, col3 = st.columns(3)
            with col1:
                X = st.number_input("X (jumlah sukses)", min_value=0, step=1, value=5, key="x_1s")
                n = st.number_input("n (ukuran sampel)", min_value=1, step=1, value=15, key="n_1s")
            with col2:
                p0 = st.number_input("p₀", min_value=0.0, max_value=1.0,
                                     value=0.5, format="%.4f", key="p0_1s")
                alpha = st.number_input("α", min_value=0.0001, max_value=0.5,
                                        value=0.05, format="%.4f", key="alpha_1s")
            with col3:
                alt = st.selectbox("Jenis Uji", ("dua sisi", "kiri", "kanan"), key="alt_1s")

            if X > n:
                st.error("X tidak boleh lebih besar dari n.")
            else:
                p_hat = X / n
                denom = math.sqrt(p0*(1-p0)/n)

                if denom == 0:
                    st.error("p₀ tidak boleh 0 atau 1.")
                else:
                    z = (p_hat - p0) / denom
                    pval = p_value(z, alt)
                    zcrit = z_critical(alpha, alt)

                    st.write(f"Proporsi sampel = {p_hat:.4f}")
                    st.write(f"Statistik Z = {z:.4f}")
                    st.write(f"P-Value = {pval:.4f}")
                    st.write(f"Nilai kritis = ({zcrit[0]:.4f}, {zcrit[1]:.4f})")

                    reject = (
                        (alt == "dua sisi" and abs(z) > zcrit[1]) or
                        (alt == "kiri" and z < zcrit[1]) or
                        (alt == "kanan" and z > zcrit[0])
                    )

                    if reject:
                        st.success(f"""
                        **Keputusan: Tolak H₀**

                        Karena statistik Z berada pada daerah penolakan  
                        (Z = {z:.4f}, nilai kritis = {zcrit[0]:.4f} s/d {zcrit[1]:.4f})

                        → Terdapat bukti signifikan bahwa **p ≠ p₀**.
                        """)
                    else:
                        st.info(f"""
                        **Keputusan: Gagal Menolak H₀**

                        Statistik Z tidak masuk daerah kritis  
                        (Z = {z:.4f}, nilai kritis = {zcrit[0]:.4f} s/d {zcrit[1]:.4f})

                        → Tidak terdapat bukti signifikan bahwa **p berbeda dari p₀**.
                        """)


# ===========================================================================================================================================================
#                           UJI PROPORSI DUA SAMPEL
# ============================================================================================================================================================

    with tab2:

        st.header("Uji Proporsi Dua Sampel")

        subA, subB, subC, subD = st.tabs(
            ["Konsep", "Rumus", "Contoh Perhitungan", "Kalkulator"]
        )

        # --------------------- KONSEP ---------------------
        with subA:
            st.subheader("Konsep Dasar")

            st.write("""
            Uji proporsi **dua sampel independen** digunakan untuk menentukan apakah
            proporsi dua populasi berbeda secara signifikan.
            
            - Dua sampel bersifat independen
            - Data berupa kategori (sukses/gagal)
            """)

            st.markdown("### Hipotesis Dua Arah")
            st.latex(r"H_0: p_1 = p_2")
            st.latex(r"H_1: p_1 \ne p_2")

            st.markdown("### Hipotesis Satu Arah (Kanan)")
            st.latex(r"H_0: p_1 \le p_2")
            st.latex(r"H_1: p_1 > p_2")

            st.markdown("### Hipotesis Satu Arah (Kiri)")
            st.latex(r"H_0: p_1 \ge p_2")
            st.latex(r"H_1: p_1 < p_2")


        # --------------------- RUMUS ---------------------
        with subB:
            st.subheader("Rumus Uji Proporsi Dua Sampel")

            st.markdown("### Proporsi Masing-masing Sampel")
            st.latex(r"\hat p_1 = \frac{X_1}{n_1}")
            st.latex(r"\hat p_2 = \frac{X_2}{n_2}")

            st.markdown("### Pooled Proportion (Karena H_0: p_1 = p_2)")
            st.latex(r"\hat p = \frac{X_1 + X_2}{n_1 + n_2}")

            st.markdown("### Statistik Uji Z")
            st.latex(
                r"""
                Z =
                \frac{\hat p_1 - \hat p_2}
                {\sqrt{\hat p(1-\hat p)
                \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
                """
            )

            st.markdown("""
            **Keterangan:**
            - **X₁, X₂** : jumlah sukses pada masing-masing sampel  
            - **n₁, n₂** : ukuran sampel pertama dan kedua  
            - **p̂₁, p̂₂** : proporsi masing-masing sampel  
            - **p̂ (pooled)** : proporsi gabungan (karena H₀ mengasumsikan p₁ = p₂)  
            - **Z** : statistik uji Z untuk dua proporsi  
            """)



        #---------------------CONTOH PERHITUNGAN-------------------
        with subC:
            st.subheader("Contoh Perhitungan Dua Sampel")

            st.write("""
            Sebuah perusahaan ingin membandingkan proporsi pelanggan puas antara dua cabang.
            - Cabang A: 20 dari 40 pelanggan puas  
            - Cabang B: 15 dari 35 pelanggan puas  
            Dengan α = 0.05, uji apakah kedua proporsi berbeda.
            """)

            # Data contoh
            X1, n1 = 20, 40
            X2, n2 = 15, 35
            p1 = X1/n1
            p2 = X2/n2
            ppool = (X1 + X2) / (n1 + n2)
            denom = math.sqrt(ppool * (1-ppool) * (1/n1 + 1/n2))
            z = (p1 - p2) / denom

            st.markdown("### Langkah 1: Hitung Proporsi Masing-masing Sampel")
            st.latex(r"\hat{p}_1 = \frac{20}{40} = 0.50")
            st.latex(r"\hat{p}_2 = \frac{15}{35} = 0.4286")

            st.markdown("### Langkah 2: Hitung Pooled Proportion")
            st.latex(
                r"""
                \hat{p} = \frac{20 + 15}{40 + 35}
                = 0.4667
                """
            )

            st.markdown("### Langkah 3: Hitung Statistik Uji Z")
            st.latex(
                r"""
                Z =
                \frac{0.50 - 0.4286}
                {\sqrt{0.4667(1-0.4667)
                \left(\frac{1}{40} + \frac{1}{35}\right)}}
                = %.4f
                """ % z
            )

            st.markdown("### Langkah 4: Hitung P-Value (Uji Dua Arah)")
            st.write("""Nilai **p-value** dihitung dari *luas daerah ekor distribusi normal standar*
            setelah nilai Z yang dihitung. Karena uji dua arah, maka daerahnya menjadi dua sisi.""")
            pval_example = 2 * (1 - stats.norm.cdf(abs(z)))
            st.latex(
                r"""
                p\text{-value}
                = 2(1 - \Phi(|Z|))
                = 2(1 - \Phi(|%.4f|))
                = %.4f
                """ % (z, pval_example)
            )

            st.markdown("### Langkah 5: Nilai Kritis (α = 0.05)")
            st.write("""
            Nilai kritis diperoleh dari **quantile distribusi normal standar**.
            Karena ini uji dua arah dengan α = 0.05:
            
            - α/2 = 0.025 pada masing-masing ekor  
            - Maka nilai kritis adalah Z pada kuantil 0.975(karena 1 − 0.025 = 0.975)
            """)
            st.latex(r"Z_{\text{krit}} = \pm 1.96")

            st.markdown("### Langkah 6: Keputusan")
            if abs(z) > 1.96:
                st.markdown(r"""
                **Tolak H₀**, karena  
                \[
                |Z| = %.4f > 1.96
                \]  
                Jadi proporsi dua cabang **berbeda secara signifikan**.
                """ % abs(z))
            else:
                st.markdown(r"""
                **Gagal Menolak H₀**, karena  
                \[
                |Z| = %.4f \le 1.96
                \]  
                Tidak ada bukti perbedaan proporsi dua cabang.
                """ % abs(z))


        # --------------------- KALKULATOR ---------------------
        with subD:
            st.subheader("Kalkulator Proporsi Dua Sampel")

            col1, col2 = st.columns(2)
            with col1:
                X1 = st.number_input("X₁", min_value=0, step=1, value=5, key="x1_2s")
                n1 = st.number_input("n₁", min_value=1, step=1, value=10, key="n1_2s")
            with col2:
                X2 = st.number_input("X₂", min_value=0, step=1, value=4, key="x2_2s")
                n2 = st.number_input("n₂", min_value=1, step=1, value=9, key="n2_2s")

            alpha2 = st.number_input("α", min_value=0.0001, max_value=0.5,
                                     value=0.05, format="%.4f", key="alpha_2s")
            alt2 = st.selectbox("Jenis Uji", ("dua sisi", "kiri", "kanan"), key="alt_2s")

            if X1 > n1 or X2 > n2:
                st.error("X₁ ≤ n₁ dan X₂ ≤ n₂ wajib.")
            else:
                p1 = X1 / n1
                p2 = X2 / n2
                ppool = (X1 + X2) / (n1 + n2)
                denom = math.sqrt(ppool * (1 - ppool) * (1/n1 + 1/n2))

                z = (p1 - p2) / denom
                pval = p_value(z, alt2)
                zcrit = z_critical(alpha2, alt2)

                st.write(f"p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}")
                st.write(f"Pooled p = {ppool:.4f}")
                st.write(f"Z = {z:.4f}")
                st.write(f"P-Value = {pval:.4f}")
                st.write(f"Nilai kritis = ({zcrit[0]:.4f}, {zcrit[1]:.4f})")

                reject = (
                    (alt2 == "dua sisi" and abs(z) > zcrit[1]) or
                    (alt2 == "kiri" and z < zcrit[1]) or
                    (alt2 == "kanan" and z > zcrit[0])
                )

                st.markdown("### Keputusan")

                if reject:
                    st.success(f"""
                    **Keputusan: Tolak H₀**

                    Karena statistik Z memenuhi kriteria penolakan  
                    (Z = {z:.4f}, nilai kritis = {zcrit[0]:.4f} s/d {zcrit[1]:.4f})

                    → Terdapat bukti signifikan bahwa **p₁ ≠ p₂**.
                    """)
                else:
                    st.info(f"""
                    **Keputusan: Gagal Menolak H₀**

                    Statistik Z tidak masuk daerah kritis  
                    (Z = {z:.4f}, nilai kritis = {zcrit[0]:.4f} s/d {zcrit[1]:.4f})

                    → Tidak ada bukti signifikan bahwa **p₁ dan p₂ berbeda**.
                    """)


#========================================================================================================================================================================
#                                 "Uji Rata-rata Satu Sampel"
#==================================================================================================================================================================================

elif menu == "Uji Rata-rata Satu Sampel":
    # ======= TITLE =======
    st.markdown(
        '<h1 class="neon-title">Uji Rata-rata 1 Sampel</h1>',
        unsafe_allow_html=True
    )
    # ======= MAIN TABS =======
    tab1, tab2, tab3, tab4 = st.tabs(["Konsep", "Rumus", "Contoh", "Hitung"])
    
    # ================================
    # TAB 1: KONSEP
    # ================================
    with tab1:
        st.markdown("## 🧠 Konsep Dasar")
        st.write("""
        Uji rata-rata satu sampel digunakan untuk membandingkan rata-rata dari satu kelompok data (sampel) dengan nilai rata-rata yang sudah diketahui 
        atau dihipotesiskan dari populasi secara keseluruhan. Tujuannya adalah untuk menentukan apakah ada perbedaan yang signifikan secara statistik 
        antara rata-rata sampel dan nilai referensi populasi tersebut. 
    
        Uji ini menggunakan **Uji Z** untuk varians yang diketahui dan **Uji T** untuk varians yang tidak diketahui.
        """)
    
        st.markdown("## 📝 Hipotesis Statistik")
    
        # ------- TABEL HIPOTESIS -------
        st.subheader("Jenis Hipotesis (1 Sampel)")
        st.markdown(r"""
    | Jenis Uji | Hipotesis Nol ($H_0$) | Hipotesis Alternatif ($H_1$) | Keterangan |
    | :---: | :---: | :---: | :--- |
    | *Dua Arah / Two tailed* | $H_0: \mu = \mu_0$ | $H_1: \mu \neq \mu_0$ | Menguji apakah rata-rata berbeda dari nilai hipotesis $\mu_0$. |
    | *Satu Arah (Kanan) / Right tail* | $H_0: \mu \le \mu_0$ | $H_1: \mu > \mu_0$ | Menguji apakah rata-rata lebih besar dari $\mu_0$. |
    | *Satu Arah (Kiri) / Left tail* | $H_0: \mu \ge \mu_0$ | $H_1: \mu < \mu_0$ | Menguji apakah rata-rata lebih kecil dari $\mu_0$. |
    """)
    
    # ================================
    # TAB 2: RUMUS
    # ================================
    with tab2:
        st.markdown("## 🎯 Uji Rata-Rata 1 Sampel dengan t Kritis / Z Kritis")
        st.markdown(r"""
    Uji rata-rata 1 sampel digunakan untuk mengetahui apakah rata-rata sampel ($\bar{X}$)
    berbeda secara signifikan dari nilai rata-rata populasi ($\mu_0$).
    
    Jenis uji bergantung pada informasi varians populasi:
    
    - Jika **varians populasi diketahui** → gunakan **Uji Z**  
    - Jika **varians populasi tidak diketahui** → gunakan **Uji t**
    """)
    
        st.markdown("### 🧩 Rumus Statistik Uji Z (varians populasi diketahui)")
        st.markdown(r"""
    $$
    Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}}
    $$
    
    **Keterangan:**
    - $\bar{X}$ : Rata-rata sampel  
    - $\sigma$ : Standar deviasi populasi  
    - $n$ : Ukuran sampel  
    - $\mu_0$ : Mean populasi yang diuji  
    """)
    
        st.info("Keputusan Uji Z: Tolak H0 jika |Z| > Z kritis (dua arah), atau sesuai arah uji.")
    
        st.markdown("### 🧩 Rumus Statistik Uji t (varians populasi tidak diketahui)")
        st.markdown(r"""
    $$
    t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}
    $$
    
    **Keterangan:**
    - $\bar{X}$ : Rata-rata sampel  
    - $s$ : Standar deviasi sampel  
    - $n$ : Ukuran sampel  
    - $df = n - 1$ : Derajat bebas  
    """)
    
        st.info("Keputusan Uji t: Tolak H0 jika |t| > t kritis (dua arah), atau sesuai arah uji.")
    
        st.markdown("## 🟥 Daerah Kritis berdasarkan Jenis Uji")
        st.markdown(r"""
    ### 🔹 **Uji Dua Arah**
    - Tolak $H_0$ jika:  
      - $|Z| > Z_{\alpha/2}$  (Uji Z)  
      - $|t| > t_{\alpha/2,\,df}$  (Uji t)
    
    ### 🔹 **Uji Satu Arah (Kanan)**
    - Tolak $H_0$ jika:  
      - $Z > Z_\alpha$  
      - $t > t_{\alpha,\,df}$
    
    ### 🔹 **Uji Satu Arah (Kiri)**
    - Tolak $H_0$ jika:  
      - $Z < -Z_\alpha$  
      - $t < -t_{\alpha,\,df}$
    """)
    
        st.success("Gunakan nilai kritis Z  atau nilai kritis t.")
    # ================================
    # TAB 3: CONTOH 
    # ================================
    with tab3:
        st.markdown("## 📘 Contoh Perhitungan Uji 1 Sampel")
    
        # Contoh Uji Z
        with st.expander("🔹 Contoh Uji Z (σ diketahui)"):
            st.markdown(r"""
    **Diketahui:**
    - $\bar{X} = 52$  
    - $\mu_0 = 50$  
    - $\sigma = 10$  
    - $n = 25$  
    - $\alpha = 0.05$ (dua arah)
    
    ### Langkah 1: Hitung Statistik Uji
    $$
    Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} 
    = \frac{52 - 50}{10 / \sqrt{25}} 
    = 1
    $$
    
    ### Langkah 2: Nilai Kritis
    $Z_{\alpha/2} = 1.96$
    
    ### Langkah 3: Keputusan
    $|Z| = 1 < 1.96 \Rightarrow$ **Gagal menolak $H_0$**
            """)
    
        # Contoh Uji t
        with st.expander("🔹 Contoh Uji t (σ tidak diketahui)"):
            st.markdown(r"""
    **Diketahui:**
    - $\bar{X} = 20$  
    - $\mu_0 = 18$  
    - $s = 5$  
    - $n = 16$  
    - $\alpha = 0.05$ (dua arah)
    
    ### Langkah 1: Hitung Statistik Uji
    $$
    t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}} 
    = \frac{20 - 18}{5 / 4} 
    = 1.6
    $$
    
    ### Langkah 2: Nilai Kritis
    $df = 15, \ t_{0.025,15} = 2.131$
    
    ### Langkah 3: Keputusan
    $|t| = 1.6 < 2.131 \Rightarrow$ **Gagal menolak $H_0$**
            """)
    # ================================
    # TAB 4: HITUNG
    # ================================
    with tab4:
        st.subheader("📌 Langkah 1: Tentukan Hipotesis")
    
        mu0 = st.number_input("Hipotesis rata-rata (μ₀)", value=0.0)
    
        alternative = st.selectbox(
            "Pilih hipotesis alternatif (H1):",
            ["≠ μ0 (Two-tailed)", "> μ0 (Right-tail)", "< μ0 (Left-tail)"]
        )
    
        alpha = st.number_input("Taraf signifikansi (α)", value=0.05, step=0.01)
    
        st.subheader("Masukkan Data Sampel")
        data_input = st.text_area(
            "Masukkan data sampel (pisahkan dengan koma)",
            "12, 15, 14, 16, 18"
        )
    
        data = np.array([float(i) for i in data_input.split(",")])
        xbar = np.mean(data)
        n = len(data)
    
        st.write(f"**Rata-rata sampel (x̄):** {xbar:.4f}")
        st.write(f"**Jumlah sampel (n):** {n}")
    
        st.subheader("📌 Langkah 2: Apakah varians/σ diketahui?")
        sigma_known = st.radio("Varians diketahui?", ["YA", "TIDAK"])
    
        # Uji Z
        if sigma_known == "YA":
            st.subheader("🟦 Menggunakan Uji Z (σ diketahui)")
            sigma = st.number_input("Masukkan standar deviasi populasi (σ)", value=1.0)
            z_stat = (xbar - mu0) / (sigma / np.sqrt(n))
    
            if alternative == "≠ μ0 (Two-tailed)":
                z_crit = norm.ppf(1 - alpha/2)
                lower = -z_crit
                upper = z_crit
            elif alternative == "> μ0 (Right-tail)":
                z_crit = norm.ppf(1 - alpha)
                lower = z_crit
                upper = None
            else:  # "< μ0"
                z_crit = norm.ppf(alpha)
                lower = None
                upper = z_crit
    
            st.write("### 📊 Hasil Uji Z:")
            st.write(f"- **Statistik Z:** {z_stat:.4f}")
            st.write(f"- **Nilai kritis Z:** {z_crit:.4f}")
    
        # Uji t
        else:
            st.subheader("🟩 Menggunakan Uji t (σ tidak diketahui)")
            s = np.std(data, ddof=1)
            df = n - 1
            st.write(f"**Standar deviasi sampel (s):** {s:.4f}")
            st.write(f"**Derajat bebas (df):** {df}")
            t_stat = (xbar - mu0) / (s / np.sqrt(n))
    
            if alternative == "≠ μ0 (Two-tailed)":
                t_crit = t.ppf(1 - alpha/2, df)
                lower = -t_crit
                upper = t_crit
            elif alternative == "> μ0 (Right-tail)":
                t_crit = t.ppf(1 - alpha, df)
                lower = t_crit
                upper = None
            else:
                t_crit = t.ppf(alpha, df)
                lower = None
                upper = t_crit
    
            st.write("### 📊 Hasil Uji t:")
            st.write(f"- **Statistik t:** {t_stat:.4f}")
            st.write(f"- **Nilai kritis t:** {t_crit:.4f}")
    
        # Keputusan
        st.subheader("📌 Keputusan Uji Hipotesis (Metode Daerah Kritis)")
        test_stat = z_stat if sigma_known == "YA" else t_stat
    
        if alternative == "≠ μ0 (Two-tailed)":
            if test_stat < lower or test_stat > upper:
                st.error("❗ **Tolak H0** — nilai berada di luar daerah kritis.")
            else:
                st.success("✔️ **Gagal menolak H0** — nilai berada di dalam daerah kritis.")
        elif alternative == "> μ0 (Right-tail)":
            if test_stat > lower:
                st.error("❗ **Tolak H0** — nilai berada di daerah kritis (kanan).")
            else:
                st.success("✔️ **Gagal menolak H0** — tidak masuk daerah kritis.")
        else:
            if test_stat < upper:
                st.error("❗ **Tolak H0** — nilai berada di daerah kritis (kiri).")
            else:
                st.success("✔️ **Gagal menolak H0** — tidak masuk daerah kritis.")
    
        # Catatan Interpretasi
        st.subheader("📝 Catatan Interpretasi")
        if alternative == "≠ μ0 (Two-tailed)":
            st.write(
                "- H0 ditolak jika |statistik uji| > nilai kritis dua arah. "
                "Jika |statistik uji| ≤ nilai kritis, H0 gagal ditolak."
            )
        elif alternative == "> μ0 (Right-tail)":
            st.write(
                "- H0 ditolak jika statistik uji > nilai kritis kanan. "
                "Jika statistik uji ≤ nilai kritis, H0 gagal ditolak."
            )
        else:
            st.write(
                "- H0 ditolak jika statistik uji < nilai kritis kiri. "
                "Jika statistik uji ≥ nilai kritis, H0 gagal ditolak."
            )
    
        st.info("Catatan: Statistik uji (Z atau t) dibandingkan dengan nilai kritis.")


     
#========================================================================================================================================================
#                                        "Uji Rata-rata 2 Sampel Independen – Varians Diketahui"
#===================================================================================================================================================================

elif menu == "Uji Rata-rata 2 Sampel Independen – Varians Diketahui":

    # ==== JUDUL UTAMA ====
    st.markdown(
        '<h1 class="neon-title">Uji Rata-rata 2 Sampel Independen - σ Varians Diketahui (Uji Z)</h1>',
        unsafe_allow_html=True
    )
    
    # ======= MAIN TABS =======
    tab1, tab2, tab3, tab4 = st.tabs(["Konsep", "Rumus", "Contoh", "Hitung"])
    
    # ================================
    # TAB 1: KONSEP
    # ================================
    with tab1:
        st.markdown("## 🧠 Konsep Dasar")
        st.write("""
        Uji dua sampel independen dengan **varians diketahui** digunakan untuk membandingkan
    dua rata-rata populasi dengan asumsi:
    
    - Dua sampel **independen**.
    - **Varians populasi** (σ¹² dan σ²²) sudah diketahui atau dianggap diketahui.
    - Ukuran sampel besar (umumnya **n > 30**).
    
    Uji ini menggunakan **distribusi normal (Z)** karena varians dianggap diketahui.
    """)
    
        st.markdown("## 📝 Hipotesis Statistik")
    
        # ------- TABEL HIPOTESIS -------
        st.subheader("Jenis Hipotesis")
        st.markdown(r"""
    | Jenis Uji | Hipotesis Nol ($H_0$) | Hipotesis Alternatif ($H_1$) | Keterangan |
    | :---: | :---: | :---: | :--- |
    | *Dua Arah* |$H_0$: $\mu_1 = \mu_2$ |$H_1$: $\mu_1 \neq \mu_2$ | Menguji perbedaan dari $\mu_1$ dan $\mu_2$. |
    | *Satu Arah (Kanan)* |$H_0$: $\mu_1 \le \mu_2$ |$H_1$: $\mu_1 > \mu_2$ | Menguji apakah $\mu_1$ lebih besar dari $\mu_2$. |
    | *Satu Arah (Kiri)* |$H_0$: $\mu_1 \ge \mu_2$ |$H_1$: $\mu_1 < \mu_2$ | Menguji apakah $\mu_1$ lebih kecil dari $\mu_2$. |
    """)
        
    # ================================
    # TAB 2: RUMUS
    # ================================
    with tab2:
        st.markdown("## 🧩 Rumus Statistik Uji Z")
    
        st.markdown(r"""
        ### Rumus Statistik Uji
        $$
        Z = \frac{\bar{X}_1 - \bar{X}_2 - (\mu_1 - \mu_2)_0}
                 {\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}}
        $$
        """)
    
        st.markdown("""
        **Keterangan variabel:**
        - 𝑥̄₁, 𝑥̄₂ : Rata-rata sampel  
        - σ₁, σ₂ : Standar deviasi populasi  
        - n₁, n₂ : Jumlah sampel  
        - (μ₁ − μ₂)₀ : Selisih populasi di bawah H₀ (biasanya 0)
        """)
    
        st.success("Daerah kritis tergantung jenis uji: dua arah, kanan, atau kiri.")
    
        # ================================
    # TAB 3: CONTOH (DENGAN PERHITUNGAN LENGKAP)
    # ================================
    with tab3:
        st.markdown("## 📘 Contoh Soal")
    
        st.write("""
        Misalkan dua metode pembelajaran ingin dibandingkan:
    
        **Sampel 1**
        - Rata-rata (x̄₁) = 80  
        - Standar deviasi populasi (σ₁) = 10  
        - Jumlah sampel (n₁) = 40  
    
        **Sampel 2**
        - Rata-rata (x̄₂) = 75  
        - Standar deviasi populasi (σ₂) = 12  
        - Jumlah sampel (n₂) = 35  
    
        Gunakan taraf signifikansi **α = 0.05**, uji **dua arah**.
        """)
    
        # ---------------------------
        # Data contoh
        # ---------------------------
        x1, x2 = 80, 75
        sigma1, sigma2 = 10, 12
        n1, n2 = 40, 35
        mu0 = 0
        alpha = 0.05
    
        # ---------------------------
        # Perhitungan langkah demi langkah
        # ---------------------------
        st.markdown("### Langkah 1: Hitung Standard Error (SE)")
        SE = math.sqrt((sigma1**2)/n1 + (sigma2**2)/n2)
        st.write(f"""
        $$
        SE = \\sqrt{{\\frac{{{sigma1}^2}}{{{n1}}} + \\frac{{{sigma2}^2}}{{{n2}}}}}
           = {SE:.4f}
        $$
        """)
    
        st.markdown("### Langkah 2: Hitung Statistik Uji Z")
        Zc = (x1 - x2 - mu0) / SE
        st.write(f"""
        $$
        Z = \\frac{{{x1} - {x2} - ({mu0})}}{{SE}}
          = {Zc:.4f}
        $$
        """)
    
        st.markdown("### Langkah 3: Hitung p-value (Uji Dua Arah)")
        pval = 2 * (1 - norm.cdf(abs(Zc)))
        st.write(f"""
        $$
        p\\text{{-value}} = 2(1 - \\Phi(|Z|))
                         = {pval:.4f}
        $$
        """)
    
        st.markdown("### Langkah 4: Hitung Nilai Kritis")
        krit = norm.ppf(1 - alpha/2)
        st.write(f"""
        $$Z_{{krit}} = \\pm Z_{{1-\\alpha/2}} = \\pm {krit:.4f}$$
        """)
    
        # ---------------------------
        # Keputusan
        # ---------------------------
        st.markdown("### 🟩 Keputusan Akhir")
    
        keputusan = "Tolak H₀" if abs(Zc) > krit else "Gagal Tolak H₀"
    
        if keputusan == "Tolak H₀":
            st.success(f"""
            **TOLAK H₀**
    
            Karena |Z| = {abs(Zc):.4f} > {krit:.4f}  
            dan p-value = {pval:.4f} < α = {alpha},
    
            → Ada bukti **perbedaan rata-rata** antara kedua metode belajar.
            """)
        else:
            st.info(f"""
            **GAGAL MENOLAK H₀**
    
            Karena |Z| = {abs(Zc):.4f} ≤ {krit:.4f}  
            dan p-value = {pval:.4f} ≥ α = {alpha},
    
            → Tidak ada bukti signifikan adanya perbedaan rata-rata.
            """)
    
    # TAB 4: HITUNG (Uji Z Dua Sampel Independen)
    # ================================
    
    def parse_and_calculate_stats(data_input):
        try:
            data_list = [float(x.strip()) for x in data_input.replace(',', ' ').split() if x.strip()]
            if len(data_list) < 2:
                return None, None, None, "Minimal diperlukan 2 data."
            return np.mean(data_list), np.var(data_list, ddof=0), len(data_list), None
        except:
            return None, None, None, "Input tidak valid. Gunakan angka dipisahkan koma/spasi."
    
    
    with tab4:
        st.subheader("Hitung Uji Z Dua Sampel Independen")
        st.markdown("Masukkan data sampel mentah. Varians dianggap **varians populasi** (σ²).")
    
        st.subheader("Sampel 1")
        data1_input = st.text_area("Masukkan Data Sampel 1 (contoh: 12, 15, 10, 14, 13)", key="d1", height=100)
    
        st.subheader("Sampel 2")
        data2_input = st.text_area("Masukkan Data Sampel 2 (contoh: 12, 15, 10, 14, 13)", key="d2", height=100)
    
        st.markdown("---")
        alpha = st.slider("Tingkat Signifikansi (α)", 0.01, 0.10, 0.05, 0.01)
    
        if st.button("Hitung", type="primary"):
            x1, var1, n1, err1 = parse_and_calculate_stats(data1_input)
            x2, var2, n2, err2 = parse_and_calculate_stats(data2_input)
    
            if err1 or err2:
                st.error(err1 or err2)
            else:
                st.success("Statistik Sampel Berhasil Dihitung.")
    
                st.subheader("📌 Statistik Sampel")
                colA, colB = st.columns(2)
                with colA:
                    st.write("**Sampel 1**")
                    st.write(f"n₁ = {n1}")
                    st.write(f"x̄₁ = {x1:.4f}")
                    st.write(f"σ₁² = {var1:.4f}")
                with colB:
                    st.write("**Sampel 2**")
                    st.write(f"n₂ = {n2}")
                    st.write(f"x̄₂ = {x2:.4f}")
                    st.write(f"σ₂² = {var2:.4f}")
    
                # ===========================
                # UJI Z
                # ===========================
                SE = math.sqrt(var1/n1 + var2/n2)
                Z_value = (x1 - x2) / SE
                p_value = 2 * (1 - norm.cdf(abs(Z_value)))
                Z_kritis = norm.ppf(1 - alpha/2)
    
                st.subheader("📊 Hasil Uji Z")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Z Hitung", f"{Z_value:.4f}")
                with col2:
                    st.metric("Z Kritis (dua sisi)", f"±{Z_kritis:.4f}")
                with col3:
                    st.metric("P-Value", f"{p_value:.4f}")
    
                st.subheader("🟩 Keputusan")
                if abs(Z_value) > Z_kritis:
                    st.error(f"TOLAK H₀ — Ada perbedaan rata-rata yang signifikan.")
                else:
                    st.info(f"GAGAL TOLAK H₀ — Tidak ada bukti perbedaan rata-rata.")
    
                st.markdown("---")
                st.subheader("Detail Tambahan")
                st.write(f"Standard Error (SE) = {SE:.4f}")
    
#===============================================================================================================================================================
#                               Uji Kesamaan Varians
#==========================================================================================================================================================================

elif menu == "Uji Kesamaan Varians":

    st.title("Uji Kesamaan Varians")
    tab1, tab2, tab3, tab4 = st.tabs(['Konsep', 'Rumus', 'Contoh', 'Statistik Uji'])
    with tab1:
        subtab1, subtab2 =st.tabs(['Konsep Dasar & Tujuan Uji', 'Hipotesis Statistik'])
        with subtab1:
            st.subheader("Konsep Dasar & Tujuan Uji")
            st.write("Menguji kesamaan varians dilakukan untuk menentukan bentuk uji statistik yang digunakan untuk menguji rata-rata dua sampel independen.") 
        with subtab2:
            st.header("Hipotesis Statistik")
            col1, col2 = st.columns(2)
            with col1: 
                st.subheader("Hipotesis 0:")
                st.markdown("**Ho: μ(A) = μ(B) atau μ(A) - μ(B) = 0**")
            with col2:
                st.subheader("Hipotesis 1:")
                st.markdown("**H1: μ(A) ≠ μ(B) atau μ(A) - μ(B) ≠ 0**")
    
    
    with tab2:
        st.header("Rumus Uji Kesamaan Varians")
        st.latex(r"""
        F_{\text{hitung}} = \frac{S^{2}_A}{ S^{2}_B}
        """)
        st.latex(r"""
        F = \frac{S_1^2}{S_2^2}
        """)
        st.latex(r"""
        \text{dengan } S_1^2 > S_2^2
        """)
        
        st.latex(r"""
        df_1 = n_1 - 1,\quad df_2 = n_2 - 1
        """)
        st.latex(r"""
        \text{Dengan } F_{\text{hitung}} > F_{\text{kritis}}\ atau \ F_{\text{hitung}} < F_{\text{kritis}} \text{=} \text{Varians Tidak Sama}
        """)
    
    with tab3:
        st.header("Contoh Perhitungan Singkat Kesamaan Varians")
        st.subheader("Soal")
        
        st.write("Sebuah perusahaan komputer memproduksi dua tipe laptop yaitu Tipe A dan Tipe B. Mereka kemudian ingin membandingkan stabilitas suhu prosesor dari kedua tipe tersebut (Dalam Celcius). ")
        st.markdown("**Data Tipe A: 55, 57, 60, 59, 58, 61, 56, 60**")
        st.write("**Data Tipe B: 50, 53, 52, 54, 55, 51, 52, 53**")
        st.write("Tentukan apakah varians suhu prosesor kedua tipe laptop tersebut sama pada taraf signifikansi α = 0.05!")
    
        st.subheader("Perhitungan Singkat")
        st.latex(r"""
        S^{2} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^{2}}{n-1}
        """)
        st.write("Akan didapatkan:")
        st.latex(r"""
        S^{2}_A = 4.5
        """)
        st.latex(r"""
        S^{2}_B = 2.57
        """)
        st.write("Dengan:")
        st.latex(r"""
        F_{\text{hitung}} = \frac{S^{2}_A}{ S^{2}_B}
        """)
        st.write("Sehingga:")
        st.latex(r"""
        F_{\text{hitung}} = 1.7499999999999998\
        F_{\text{kritis atas}} = 4.994909219063238\
        F_{\text{kritis bawah}} = 0.20020383877718267
        """)
        
        
        st.subheader("Kriteria Uji")
        st.latex(r"""
        \text{Tolak } H_0 \text{ jika } F_{\text{hitung}} > F_{\text{kritis atas}} \text{ atau } F_{\text{hitung}} < F_{\text{kritis bawah}}
        """)
    
        st.subheader("Keputusan")
        st.latex(r"\text{Gagal menolak } H_0 \text{: Varians sama}")
        
    
    
    with tab4:
        st.subheader("Statistik Uji Kesamaan Varians")
        def ke_df(text):
            try:
                angka = [float(x.strip()) for x in text.split(",")]
                return pd.DataFrame({"Nilai": angka})
            except ValueError:
                st.error("Pastikan input sudah valid dan dipisahkan koma")
        
        st.subheader("Input Data")
        data1 = st.text_area("Masukkan data Kelompok 1 (pisahkan dengan koma):")
        data2 = st.text_area("Masukkan data Kelompok 2 (pisahkan dengan koma):")
        
        
        df1 = ke_df(data1)
        df2 = ke_df(data2)
            
        st.subheader("Uji Kesamaan Varians | α = 5%")
        if st.button("Uji Kesamaan Varians"):
            s1 = df1["Nilai"].var()
            s2 = df2["Nilai"].var()
            n1 = len(df1)
            n2 = len(df2)
            a = 0.05
            if s1 >= s2:
                Fhit = s1 / s2
                dof1 = n1 - 1
                dof2 = n2 - 1
            else:
                Fhit = s2 / s1
                dof1 = n2 - 1
                dof2 = n1 - 1
            Fatas = f.ppf(1 - a/2, dof1, dof2)
            Fbawah = f.ppf(a/2, dof1, dof2)
    
    
            st.info(f"**F = {Fhit}**")
            st.info(f"**F-Atas = {Fatas}**")
            st.info(f"**F-Bawah = {Fbawah}**")
            if Fhit < Fbawah or Fhit > Fatas:
                st.error("**Tolak H0: Varians tidak sama.**")
            else:
                st.success("**Gagal menolak H0: Varians sama.**")

#======================================================================================================================================
#                                       Uji Rata-rata 2 Sampel Independen – Varians Sama
#====================================================================================================================

elif menu == "Uji Rata-rata 2 Sampel Independen – Varians Sama":
    # Fungsi untuk memparsing input string data mentah dan menghitung statistik
    def parse_and_calculate_stats(data_input):
        data_input = data_input.replace(',', ' ').replace(';', ' ') # Ganti koma/titik koma dengan spasi
        
        try:
            # Filter dan konversi string ke float, menggunakan spasi sebagai delimiter
            data_list = [float(x) for x in data_input.split() if x]
        except ValueError:
            return None, None, 0, "Error: Pastikan semua input adalah angka yang valid."
    
        n = len(data_list)
        if n < 2:
            return None, None, 0, "Error: Ukuran sampel minimal harus 2 untuk setiap kelompok."
    
        data = np.array(data_list)
        x_bar = np.mean(data)
        # Hitung Varians Sampel (s^2) menggunakan ddof=1 (pembagi n-1)
        s_sq = np.var(data, ddof=1)
        
        return x_bar, s_sq, n, None
    
    # ======= TITLE =======
    st.markdown(
        '<h1 class="neon-title">Uji Rata-rata 2 Sampel Independen – Varians Sama (Pooled t-test)</h1>',
        unsafe_allow_html=True
    )
    
    st.write("---")
    
    # ======= MAIN TABS =======
    tab1, tab2, tab3, tab4 = st.tabs(["Konsep", "Rumus", "Contoh", "Hitung"])
    
    # ================================
    # TAB 1: KONSEP
    # ================================
    with tab1:
        st.markdown("## 🧠 Konsep Dasar")
        st.write("""
        Uji t dua sampel independen (pooled) digunakan untuk mengetahui apakah terdapat perbedaan rata-rata dua kelompok yang tidak saling berhubungan, dengan asumsi bahwa varians populasi kedua kelompok tersebut adalah sama. 
        Uji ini mensyaratkan 
        (1) data berdistribusi normal dan 
        (2) asumsi varians sama
        
    """)
    
        st.markdown("## 📝 Hipotesis Statistik")
    
        # ------- TABEL HIPOTESIS -------
        st.subheader("Jenis Hipotesis")
        st.markdown(r"""
    | Jenis Uji | Hipotesis Nol ($H_0$) | Hipotesis Alternatif ($H_1$) | Keterangan |
    | :---: | :---: | :---: | :--- |
    | *Dua Arah* |$H_0$: $\mu_1 = \mu_2$ |$H_1$: $\mu_1 \neq \mu_2$ | Menguji perbedaan dari $\mu_1$ dan $\mu_2$. |
    | *Satu Arah (Kanan)* |$H_0$: $\mu_1 \le \mu_2$ |$H_1$: $\mu_1 > \mu_2$ | Menguji apakah $\mu_1$ lebih besar dari $\mu_2$. |
    | *Satu Arah (Kiri)* |$H_0$: $\mu_1 \ge \mu_2$ |$H_1$: $\mu_1 < \mu_2$ | Menguji apakah $\mu_1$ lebih kecil dari $\mu_2$. |
    """)
        
    # ================================
    # TAB 2: RUMUS
    # ================================
    with tab2:
        st.markdown("## 🧩 Rumus Statistik Uji t") # Mengganti Z menjadi t
        st.markdown(r"""
        ###  a. Varians Gabungan (Pooled Variance, $S_p^2$)
        $$
        S_p^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}
        $$
        """)
        
        st.markdown("""
        **Keterangan variabel:**
        - $n_1, n_2$ : Ukuran sampel 1 dan 2  
        - $S_1^2, S_2^2$ : Varians sampel (bukan simpangan baku)
        """)
    
        st.markdown(r"""
        ### b. Statistik Uji
        $$
        t = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}
            {S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
        $$
        """)
    
        st.markdown("""
        **Keterangan variabel:**
        - $\bar{X}_1, \bar{X}_2$ : Rata-rata sampel  
        - $(\mu_1 - \mu_2)$: Perbedaan rata-rata populasi yang dihipotesiskan (Biasanya 0)  
        - $n_1, n_2$ : Jumlah sampel   
        - $S_p$ : Simpangan baku gabungan ($S_p = \sqrt{S_p^2}$)
        """)
    
        st.success("Daerah kritis tergantung jenis uji: dua arah, kanan, atau kiri. Derajat kebebasan ($df$) adalah $n_1 + n_2 - 2$.")
    
    # ================================
    # TAB 3: CONTOH (DENGAN PERHITUNGAN LENGKAP)
    # ================================
    with tab3:
        st.markdown("## 📘 Contoh Soal")
        alpha = 0.05
        x1 = 78.5  # Rata-rata sampel 1
        s1 = 8.0   # Simpangan baku sampel 1
        n1 = 35    # Ukuran sampel 1
        x2 = 74.2  # Rata-rata sampel 2
        s2 = 7.5   # Simpangan baku sampel 2
        n2 = 30    # Ukuran sampel 2
        mu0 = 0    # Perbedaan rata-rata yang dihipotesiskan (H0: μ1 - μ2 = 0)
    
        st.header("🔬 Hasil Uji t Dua Sampel (Varians Sama)")
        st.markdown(f"**Data Contoh:**")
        st.markdown(f"* Kelompok 1: $\\bar{{x}}_1 = {x1}$, $s_1 = {s1}$, $n_1 = {n1}$")
        st.markdown(f"* Kelompok 2: $\\bar{{x}}_2 = {x2}$, $s_2 = {s2}$, $n_2 = {n2}$")
        st.markdown(f"**Asumsi:** Varians Populasi Dianggap Sama ($\sigma_1^2 = \sigma_2^2$).")
        st.markdown(f"**Tingkat Signifikansi ($\\alpha$):** {alpha}")
        st.markdown("---")
    
    # ---------------------------
    # Perhitungan Uji t Pooled
    # ---------------------------
    
    # Langkah A: Hitung Derajat Kebebasan (df)
        df = n1 + n2 - 2
        st.markdown("### Langkah 1: Hitung Derajat Kebebasan ($df$)")
        st.write(f"$$df = n_1 + n_2 - 2 = {n1} + {n2} - 2 = {df}$$")
    
    # Langkah B: Hitung Pooled Variance (s_p^2)
        s1_sq = s1**2
        s2_sq = s2**2
        st.markdown("### Langkah 2: Hitung Pooled Variance ($s_p^2$)")
        sp2 = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / df
        st.write(f"""
        $$
        s_p^2 = \\frac{{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}}{{n_1 + n_2 - 2}}
              = \\frac{{({n1} - 1)({s1_sq:.2f}) + ({n2} - 1)({s2_sq:.2f})}}{{{df}}}
              = {sp2:.4f}
        $$
        """)
    
    # Langkah C: Hitung Standard Error (SE) Pooled
        st.markdown("### Langkah 3: Hitung Standard Error (SE) Pooled")
        SE = math.sqrt(sp2 * (1/n1 + 1/n2))
        st.write(f"""
        $$
        SE = \\sqrt{{s_p^2 \\left(\\frac{{1}}{{n_1}} + \\frac{{1}}{{n_2}}\\right)}}
            = \\sqrt{{{sp2:.4f} \\left(\\frac{{1}}{{{n1}}} + \\frac{{1}}{{{n2}}}\\right)}}
            = {SE:.4f}
        $$
        """)
    
    # Langkah D: Hitung Statistik Uji t
        st.markdown("### Langkah 4: Hitung Statistik Uji t")
        tc = (x1 - x2 - mu0) / SE
        st.write(f"""
        $$
        t = \\frac{{(\\bar{{x}}_1 - \\bar{{x}}_2) - \\mu_0}}{{SE}}
          = \\frac{{({x1} - {x2}) - ({mu0})}}{{{SE:.4f}}}
          = {tc:.4f}
        $$
        """)
    
    # Langkah E: Hitung p-value (Uji Dua Arah)
        st.markdown("### Langkah 5: Hitung p-value (Uji Dua Arah)")
    # Menggunakan distribusi t dengan derajat kebebasan 'df'
        pval = 2 * (1 - stats.t.cdf(abs(tc), df))
        st.write(f"""
        $$
        p\\text{{-value}} = 2 \\times P(t > |t_c|) \\text{{ dengan }} df={df}
                          = {pval:.4f}
        $$
        """)
    
    # Langkah F: Hitung Nilai Kritis
        st.markdown("### Langkah 6: Hitung Nilai Kritis")
    # Menggunakan distribusi t dengan derajat kebebasan 'df'
        krit = stats.t.ppf(1 - alpha/2, df)
        st.write(f"""
        $$t_{{krit}} = \\pm t_{{1-\\alpha/2, df={df}}} = \\pm {krit:.4f}$$
        """)
    
    # ---------------------------
    # Keputusan
    # ---------------------------
        st.markdown("---")
        st.markdown("### 🟩 Keputusan Akhir")
    
        keputusan = "Tolak H₀" if abs(tc) > krit else "Gagal Tolak H₀"
    
        if keputusan == "Tolak H₀":
            st.success(f"""
            **TOLAK H₀** (Terdapat perbedaan yang signifikan)
    
            Karena |t| = {abs(tc):.4f} > {krit:.4f}
            dan p-value = {pval:.4f} < \\alpha = {alpha},
    
            → **Ada bukti** perbedaan rata-rata antara kedua populasi.
            """)
        else:
            st.info(f"""
            **GAGAL MENOLAK H₀** (Tidak terdapat perbedaan yang signifikan)
    
            Karena |t| = {abs(tc):.4f} $\\le$ {krit:.4f}
            dan p-value = {pval:.4f} $\\ge$ \\alpha = {alpha},
    
            → **Tidak ada bukti signifikan** adanya perbedaan rata-rata.
            """)
        
    def pooled_two_sample_t_test(x1_bar, s1_sq, n1, x2_bar, s2_sq, n2, alpha=0.05):
        """
        Melakukan perhitungan Uji t Dua Sampel dengan Asumsi Varians Sama (Pooled).
        """
        # 4. Hitung Derajat Kebebasan (df)
        df = int(n1 + n2 - 2)
        
        # Validasi df
        if df <= 0:
            return "Error: Derajat Kebebasan (df) tidak valid.", None, None, None, None, None
    
        # 1. Hitung Varians Gabungan (Pooled Variance, s_p^2)
        numerator_sp = ((n1 - 1) * s1_sq) + ((n2 - 1) * s2_sq)
        s_p_sq = numerator_sp / df
        
        # 2. Hitung Standard Error (SE)
        SE = np.sqrt(s_p_sq * ( (1/n1) + (1/n2) ))
        
        # 3. Hitung Nilai t (Test Statistic)
        t_value = (x1_bar - x2_bar) / SE
        
        # 5. Hitung P-value (untuk uji dua sisi)
        p_value = stats.t.sf(np.abs(t_value), df) * 2
        
        # 6. Hitung Nilai Kritis (untuk uji dua sisi)
        t_krit = stats.t.ppf(1 - alpha/2, df)
        
        return t_value, df, s_p_sq, SE, p_value, t_krit
        
    with tab4:
        st.subheader("Statistik Uji t dari Data Mentah")
        st.markdown("Masukkan data mentah (angka) dari masing-masing sampel, dipisahkan oleh **koma atau spasi**.")
    
        # Input Sampel 1
        st.subheader("Sampel 1")
        data1_input = st.text_area(
            "Masukkan Data Sampel 1 (contoh: 12, 15, 10, 14, 13)",
            key="data1",
            height=100
        )
    
        # Input Sampel 2
        st.subheader("Sampel 2")
        data2_input = st.text_area(
            "Masukkan Data Sampel 2 (contoh: 18, 17, 19, 15, 20)",
            key="data2",
            height=100
        )
    
        # Input Tingkat Signifikansi
        st.markdown("---")
        alpha = st.slider("Tingkat Signifikansi ($\\alpha$)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    
        # Tombol Perhitungan
        if st.button("Hitung", type="primary"):
            
            # 1. Hitung Statistik dari Data Mentah
            x1_bar, s1_sq, n1, error1 = parse_and_calculate_stats(data1_input)
            x2_bar, s2_sq, n2, error2 = parse_and_calculate_stats(data2_input)
            
            # Cek Error Input
            if error1 or error2:
                st.error(error1 or error2)
            elif n1 < 2 or n2 < 2:
                st.error("Ukuran sampel minimal harus 2 untuk setiap kelompok.")
            else:
                st.success("Statistik Sampel Berhasil Dihitung. Melakukan Uji t...")
                
                # Tampilkan Statistik Sampel yang Dihitung
                st.subheader("Statistik Ringkasan yang Dihitung")
                colA, colB = st.columns(2)
                with colA:
                    st.write("*Kelompok 1*")
                    st.markdown(f"* $n_1$: *{n1}*")
                    st.markdown(f"* $\\bar{{x}}_1$: *{x1_bar:.4f}*")
                    st.markdown(f"* $s_1^2$ (Varians Sampel): *{s1_sq:.4f}*")
                with colB:
                    st.write("*Kelompok 2*")
                    st.markdown(f"* $n_2$: *{n2}*")
                    st.markdown(f"* $\\bar{{x}}_2$: *{x2_bar:.4f}*")
                    st.markdown(f"* $s_2^2$ (Varians Sampel): *{s2_sq:.4f}*")
                    
                # --- Detail Perhitungan ---
                st.markdown("---")
                
                # 2. Lakukan Uji t
                result = pooled_two_sample_t_test(x1_bar, s1_sq, n1, x2_bar, s2_sq, n2, alpha)
                
                if isinstance(result, str):
                     st.error(result)
                else:
                    t_value, df, s_p_sq, SE, p_value, t_krit = result
                
                    # --- Tampilkan Hasil Utama ---
                    st.subheader("Hasil Uji-t Pooled (Dua Sisi)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Statistik Uji-t ($t$)", f"{t_value:.4f}")
                    with col2:
                        st.metric("Derajat Kebebasan ($df$)", df)
                    with col3:
                        st.metric("P-Value (Dua Sisi)", f"{p_value:.4f}")
    
                    # --- Kesimpulan ---
                    st.subheader(f"Kesimpulan Statistik (dengan $\\alpha = {alpha}$)")
                    
                    if p_value < alpha:
                        st.error(f"""
                        *TOLAK Hipotesis Nol ($H_0$)*
                        Karena P-Value ({p_value:.4f}) *<* $\\alpha$ ({alpha:.2f}), terdapat *bukti signifikan*
                        bahwa ada perbedaan rata-rata yang signifikan secara statistik antara kedua kelompok.
                        """)
                    else:
                        st.info(f"""
                        *GAGAL MENOLAK Hipotesis Nol ($H_0$)*
                        Karena P-Value ({p_value:.4f}) *>* $\\alpha$ ({alpha:.2f}), *tidak cukup bukti signifikan*
                        bahwa terdapat perbedaan rata-rata yang signifikan secara statistik antara kedua kelompok.
                        """)
                        
                    st.markdown("---")
                    
                    # --- Detail Perhitungan ---
                    st.subheader("Detail Tambahan")
                    st.write(f"*Varians Gabungan ($S_p^2$):* {s_p_sq:.4f}")
                    st.write(f"*Standard Error ($\text{{SE}}$):* {SE:.4f}")
                    st.write(f"*Nilai Kritis ($t_{{krit}}$ Dua Sisi):* $\\pm {t_krit:.4f}$")

#=================================================================================================================================
#                               "Uji Rata-rata 2 Sampel Independen – Varians Tidak Sama"
#==================================================================================================================================
elif menu == "Uji Rata-rata 2 Sampel Independen – Varians Tidak Sama":

    # ==== JUDUL UTAMA ====
    st.markdown(
        '<h1 class="neon-title">Uji rata-rata 2 Sampel Independen - Varians Tidak Sama (Welch t-test)</h1>',
        unsafe_allow_html=True
    )
    
    # ======= MAIN TABS =======
    tab1, tab2, tab3, tab4 = st.tabs(["Konsep", "Rumus", "Contoh", "Hitung"])
    
    # ================================
    # TAB 1: KONSEP
    # ================================
    with tab1:
        st.markdown("## 🧠 Konsep Dasar")
        st.write("""
       Uji-t Welch adalah prosedur statistik yang kuat (robust) untuk membandingkan rata-rata dua kelompok independen tanpa mengharuskan varians populasi kedua kelompok tersebut sama ($\sigma_1^2 = \sigma_2^2$).
       Uji Welch digunakan dalam asumsi:
       
    - Dua Sampel Independen: Data berasal dari dua kelompok yang tidak saling berhubungan (misalnya, dua kelompok perlakuan yang berbeda).
    - Varians Populasi Tidak Sama: Varians sampel ($s_1^2$ dan $s_2^2$) yang diamati sangat berbeda, mengindikasikan bahwa varians populasi ($\sigma_1^2$ dan $\sigma_2^2$) kemungkinan besar tidak sama.
    - Ukuran Sampel Tidak Sama: Uji Welch sangat disarankan ketika varians tidak sama dan ukuran sampel ($n_1$ dan $n_2$) berbeda secara signifikan.
    """)
    
        st.markdown("## 📝 Hipotesis Statistik")
    
        # ------- TABEL HIPOTESIS -------
        st.subheader("Jenis Hipotesis")
        st.markdown(r"""
    | Jenis Uji | Hipotesis Nol ($H_0$) | Hipotesis Alternatif ($H_1$) | Keterangan |
    | :---: | :---: | :---: | :--- |
    | *Dua Arah* |$H_0$: $\mu_1 = \mu_2$ |$H_1$: $\mu_1 \neq \mu_2$ | Menguji perbedaan dari $\mu_1$ dan $\mu_2$. |
    | *Satu Arah (Kanan)* |$H_0$: $\mu_1 \le \mu_2$ |$H_1$: $\mu_1 > \mu_2$ | Menguji apakah $\mu_1$ lebih besar dari $\mu_2$. |
    | *Satu Arah (Kiri)* |$H_0$: $\mu_1 \ge \mu_2$ |$H_1$: $\mu_1 < \mu_2$ | Menguji apakah $\mu_1$ lebih kecil dari $\mu_2$. |
    """)
        
    # ================================
    # TAB 2: RUMUS
    # ================================
    with tab2:
        st.markdown("## 🧩 Rumus Statistik Welch t-test")
    
        st.markdown(r"""
        ### Rumus Statistik Uji
        $$
        t = \frac{\bar{X}_1 - \bar{X}_2}
                 {\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
        $$
        """)
    
        st.markdown(r"""
        ### Derajat Kebebasan
        $$
        df = 
        \frac{
            \left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2
        }
        {
            \frac{ \left( \frac{s_1^2}{n_1} \right)^2 }{n_1 - 1}
            +
            \frac{ \left( \frac{s_2^2}{n_2} \right)^2 }{n_2 - 1}
        }
        $$
        """)
    
        st.markdown("""
        **Keterangan variabel:**
        - 𝑥̄₁, 𝑥̄₂ : Rata-rata sampel  
        - s₁², s₂² : Varians sampel  
        - n₁, n₂ : Jumlah data pada masing-masing sampel  
        - df : derajat kebebasan (tidak harus bilangan bulat)
        """)
    
        st.success("Welch t-test digunakan ketika varians antar sampel **tidak sama** dan ukuran sampel berbeda.")
    
        # ================================
    # TAB 3: CONTOH (DENGAN PERHITUNGAN LENGKAP)
    # ================================
    with tab3:
        st.markdown("## 📘 Contoh Soal (Welch t-test)")
    
        st.write("""
        Misalkan dua metode pembelajaran ingin dibandingkan:
    
        **Sampel 1**
        - Rata-rata (x̄₁) = 80  
        - Standar deviasi sampel (s₁) = 10  
        - Jumlah sampel (n₁) = 40  
    
        **Sampel 2**
        - Rata-rata (x̄₂) = 75  
        - Standar deviasi sampel (s₂) = 12  
        - Jumlah sampel (n₂) = 35  
    
        Gunakan taraf signifikansi **α = 0.05**, uji **dua arah**.
        """)
    
        # ---------------------------
        # Data contoh
        # ---------------------------
        x1, x2 = 80, 75
        s1, s2 = 10, 12
        n1, n2 = 40, 35
        alpha = 0.05
    
        # ---------------------------
        # Perhitungan langkah demi langkah
        # ---------------------------
    
        st.markdown("### Langkah 1: Hitung Standard Error (SE)")
    
        SE = math.sqrt((s1**2)/n1 + (s2**2)/n2)
    
        st.write(f"""
        $$
        SE = \\sqrt{{\\frac{{{s1}^2}}{{{n1}}} + \\frac{{{s2}^2}}{{{n2}}}}}
           = {SE:.4f}
        $$
        """)
    
        st.markdown("### Langkah 2: Hitung Statistik Uji t (Welch)")
        t_stat = (x1 - x2) / SE
    
        st.write(f"""
        $$
        t = \\frac{{{x1} - {x2}}}{{SE}}
          = {t_stat:.4f}
        $$
        """)
    
        st.markdown("### Langkah 3: Hitung Derajat Kebebasan (df) – Rumus Welch")
    
        df_num = ( (s1**2 / n1) + (s2**2 / n2) )**2
        df_den = ((s1**2 / n1)**2) / (n1 - 1) + ((s2**2 / n2)**2) / (n2 - 1)
        df = df_num / df_den
    
        st.write(f"""
        $$
        df = {df:.4f}
        $$
        """)
    
        st.markdown("### Langkah 4: Hitung p-value (Uji Dua Arah)")
        pval = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
        st.write(f"""
        $$
        p\\text{{-value}} = 2(1 - t(|t|, df))
                         = {pval:.4f}
        $$
        """)
    
        st.markdown("### Langkah 5: Hitung Nilai Kritis t")
        t_krit = stats.t.ppf(1 - alpha/2, df)
    
        st.write(f"""
        $$t_{{krit}} = \\pm t_{{1-\\alpha/2, df}} = \\pm {t_krit:.4f}$$
        """)
    
        # ---------------------------
        # Keputusan
        # ---------------------------
        st.markdown("### 🟩 Keputusan Akhir (Welch t-test)")
    
        keputusan = "Tolak H₀" if abs(t_stat) > t_krit else "Gagal Tolak H₀"
    
        if keputusan == "Tolak H₀":
            st.success(f"""
            **TOLAK H₀**
    
            Karena |t| = {abs(t_stat):.4f} > {t_krit:.4f}  
            dan p-value = {pval:.4f} < α = {alpha},
    
            → Ada bukti **perbedaan rata-rata** antara kedua metode pembelajaran.
            """)
        else:
            st.info(f"""
            **GAGAL MENOLAK H₀**
    
            Karena |t| = {abs(t_stat):.4f} ≤ {t_krit:.4f}  
            dan p-value = {pval:.4f} ≥ α = {alpha},
    
            → Tidak ada bukti signifikan adanya perbedaan rata-rata.
            """)
    # TAB 4: HITUNG (Uji Z Dua Sampel Independen)
    # ================================
    
    def parse_and_calculate_stats(data_input):
        try:
            data_list = [float(x.strip()) for x in data_input.replace(',', ' ').split() if x.strip()]
            if len(data_list) < 2:
                return None, None, None, "Minimal diperlukan 2 data."
            return np.mean(data_list), np.var(data_list, ddof=0), len(data_list), None
        except:
            return None, None, None, "Input tidak valid. Gunakan angka dipisahkan koma/spasi."
    
    
    with tab4:
        st.subheader("Hitung — Uji rata-rata 2 Sampel Independen (Welch t-test)")
        st.markdown("Masukkan data sampel mentah. Varians dianggap **tidak sama**, sehingga digunakan *Welch t-test*.")
    
    
        st.subheader("Sampel 1")
        data1_input = st.text_area("Masukkan Data Sampel 1 (contoh: 12, 15, 10, 14, 13)", key="d1", height=100)
    
        st.subheader("Sampel 2")
        data2_input = st.text_area("Masukkan Data Sampel 2 (contoh: 18, 17, 19, 15, 20)", key="d2", height=100)
    
        st.markdown("---")
        alpha = st.slider("Tingkat Signifikansi (α)", 0.01, 0.10, 0.05, 0.01)
    
        if st.button("Hitung", type="primary"):
            x1, var1, n1, err1 = parse_and_calculate_stats(data1_input)
            x2, var2, n2, err2 = parse_and_calculate_stats(data2_input)
    
            if err1 or err2:
                st.error(err1 or err2)
            else:
                st.success("Statistik Sampel Berhasil Dihitung.")
    
                st.subheader("📌 Statistik Sampel")
                colA, colB = st.columns(2)
                with colA:
                    st.write("**Sampel 1**")
                    st.write(f"n₁ = {n1}")
                    st.write(f"x̄₁ = {x1:.4f}")
                    st.write(f"σ₁² = {var1:.4f}")
                with colB:
                    st.write("**Sampel 2**")
                    st.write(f"n₂ = {n2}")
                    st.write(f"x̄₂ = {x2:.4f}")
                    st.write(f"σ₂² = {var2:.4f}")
    
                # ===========================
                # UJI Z
                # ===========================
                SE = math.sqrt(var1/n1 + var2/n2)
                Z_value = (x1 - x2) / SE
                p_value = 2 * (1 - norm.cdf(abs(Z_value)))
                Z_kritis = norm.ppf(1 - alpha/2)
    
                st.subheader("📊 Hasil Uji Z")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Z Hitung", f"{Z_value:.4f}")
                with col2:
                    st.metric("Z Kritis (dua sisi)", f"±{Z_kritis:.4f}")
                with col3:
                    st.metric("P-Value", f"{p_value:.4f}")
    
                st.subheader("🟩 Keputusan")
                if abs(Z_value) > Z_kritis:
                    st.error(f"TOLAK H₀ — Ada perbedaan rata-rata yang signifikan.")
                else:
                    st.info(f"GAGAL TOLAK H₀ — Tidak ada bukti perbedaan rata-rata.")
    
                st.markdown("---")
                st.subheader("Detail Tambahan")
                st.write(f"Standard Error (SE) = {SE:.4f}")

#============================================================================================================
#                                 "Uji Rata-rata 2 Sampel Dependen"
#===============================================================================================================
elif menu == "Uji Rata-rata 2 Sampel Dependen":

    # ==== JUDUL UTAMA ====
    st.markdown(
        '<h1 class="neon-title">Uji Rata-rata 2 Sampel Dependen (Paired Sample t-test)</h1>',
        unsafe_allow_html=True
    )
    
    # ======= MAIN TABS =======
    tab1, tab2, tab3, tab4 = st.tabs(["Konsep", "Rumus", "Contoh", "Hitung"])
    
    # ================================
    # TAB 1: KONSEP
    # ================================
    with tab1:
        st.markdown("## 🧠 Konsep Dasar")
        st.write("""
        Uji dua sampel dependen digunakan untuk melihat apakah terdapat perbedaan rata-rata 
        yang signifikan antara dua populasi yang saling berkaitan.
        - Dua sampel berpasangan.
        - Pengukuran berulang (sebelum/sesudah).
        - Menghilangkan variasi di antara para subjek.
        - Kedua populasi berdistribusi normal.
    """)
    
        st.markdown("## 📝 Hipotesis Statistik")
    
        # ------- TABEL HIPOTESIS -------
        st.subheader("Jenis Hipotesis")
        st.markdown(r"""
    | Jenis Uji | Hipotesis Nol ($H_0$) | Hipotesis Alternatif ($H_1$) | Keterangan |
    | :---: | :---: | :---: | :--- |
    | *Dua Arah* |$H_0$: $\mu_1 = \mu_2$ |$H_1$: $\mu_1 \neq \mu_2$ | Menguji perbedaan dari $\mu_1$ dan $\mu_2$. |
    | *Satu Arah (Kanan)* |$H_0$: $\mu_1 \le \mu_2$ |$H_1$: $\mu_1 > \mu_2$ | Menguji apakah $\mu_1$ lebih besar dari $\mu_2$. |
    | *Satu Arah (Kiri)* |$H_0$: $\mu_1 \ge \mu_2$ |$H_1$: $\mu_1 < \mu_2$ | Menguji apakah $\mu_1$ lebih kecil dari $\mu_2$. |
    """)
        
    # ================================
    # TAB 2: RUMUS
    # ================================
    with tab2:
        st.markdown("## 🧩 Rumus Statistik Uji t")
    
        st.markdown(r"""
        ### Rumus Statistik Uji
        $$
        t = \frac{\bar{D} - \mu_0}
                 {\frac{S_D}{\sqrt{n}}}
    
        $$
        """)
    
        st.markdown(r"""
        $$
        S_D = \sqrt{\frac{\sum_{i=1}^{n} (D_i - \bar{D})^2}{n - 1}}
    
        $$
        """)
        
        st.markdown("""
        **Keterangan variabel:**
        - $\overline{D}$ : Rata-rata selisih $X_1$ dan $X_2$
        - $\mu_0$ : Nilai rata-rata yang diujikan
        - $S_D$ : Standar deviasi
        - $D$ : Selisih antara $X_1$ dan $X_2$
        - $df$ : Derajat bebas ($df = n - 1$)
        - $n$ : Banyak data selisih
        """)
    
        st.success("Daerah kritis tergantung jenis uji : dua arah, kanan, atau kiri.")
    
        # ================================
    # TAB 3: CONTOH (DENGAN PERHITUNGAN LENGKAP)
    # ================================
    with tab3:
        st.markdown("## 📘 Contoh Soal")
    
        st.write("""
        Misalkan kita ingin membandingkan mengetahui apakah program pelatihan keterampilan kerja 
        yang diberikan kepada karyawannya berpengaruh terhadap peningkatan produktivitas :
    
        - **Sampel 1** = [32, 42, 33, 29, 24]
        - **Sampel 2** = [38, 48, 22, 25, 25]
        - **Selisih**  = [-6, -6, 11, 4, -1]
    
        Gunakan taraf signifikansi **α = 0.05**, uji **dua arah**.
        """)
    
        # ---------------------------
        # Data contoh
        # ---------------------------
        sampel1 = [32, 42, 33, 29, 24]
        sampel2 = [38, 48, 22, 25, 25]
        mu0 = 0
        alpha = 0.05
        
        st.markdown("### Langkah 1: Hitung Rata-Rata dari Selisih Kedua Sampel")
        
        selisih = np.array(sampel1) - np.array(sampel2)
        sum_selisih = np.sum(selisih)
        n = len(selisih)
        dbar = sum_selisih / n
        
        st.write(f"""
        $$
        \overline{{D}} = \\frac{{\\sum_{{i=1}}^{{n}} D_i}}{{n}}
                       = \\frac{{{sum_selisih}}}{{{n}}} = {dbar}
        $$
        """)
    
        st.markdown("### Langkah 2: Hitung Standar Deviasi Selisih Kedua Sampel")
    
        y = selisih - dbar
        sum_y = np.sum(y**2)
        sd = math.sqrt(sum_y/(n-1))
        
        st.write(f"""
        $$
        S_D = \\sqrt{{\\frac{{\\sum_{{i=1}}^{{n}} (D_i - \\overline{{D}})^2}}{{n - 1}}}}
            = \\sqrt{{\\frac{{({sum_y:.4f})}} {{{n}- 1}}}} 
            = {sd:.4f}
        $$
        """)
    
        st.markdown("### Langkah 3: Hitung Standard Error")
        SE = sd/math.sqrt(n)
        st.write(f"""
        $$
        SE = \\frac{{S_D}}{{\\sqrt{{n}}}} = \\frac{{{sd:.4f}}}{{\\sqrt{n}}}
           = {SE:.4f}
        $$
        """)
        
        st.markdown("### Langkah 4: Hitung Statistik Uji t")
        t_hitung = dbar / SE
        st.write(f"""
        $$
        t_{{hitung}} = \\frac{{\\overline{{D}}}}{{SE}}
                     = \\frac{{{dbar}}}{{{SE:.4f}}}
                     = {t_hitung:.4f}
        $$
        """)
    
        st.markdown("### Langkah 5: Hitung Nilai t Tabel")
        df = len(selisih)-1
        t_tabel = t.ppf(1 - alpha/2, df)
        st.write(f"""
        $$
        t_{{tabel}} = t_{{1-\\frac{{\\alpha}}{{2}}}} = {t_tabel:.4f}
        $$
        """)
    
        # ---------------------------
        # Keputusan
        # ---------------------------
        st.markdown("### 🟩 Keputusan Akhir")
    
        keputusan = "Tolak H₀" if abs(t_hitung) > t_tabel else "Gagal Tolak H₀"
    
        if keputusan == "Tolak H₀":
            st.success(f"""
            **TOLAK H₀**
    
            Karena |t| = {abs(t_hitung):.4f} > {t_tabel:.4f} 
    
            → Ada bukti **perbedaan rata-rata** antara kedua metode belajar.
            """)
        else:
            st.info(f"""
            **GAGAL MENOLAK H₀**
    
            Karena |t| = {abs(t_hitung):.4f} ≤ {t_tabel:.4f}  
    
            → Tidak ada bukti signifikan adanya perbedaan rata-rata.
            """)
    
    # ================================
    # TAB 4: HITUNG
    # ================================
    def list_data(data_input):
        data_list = [float(x.strip()) for x in data_input.replace(",", " ").split()]
        return data_list
        
    def parse_and_calculate_stats(data):
        try:
            xbar = np.mean(data)
            n = len(data)
            return xbar, n, None
        except Exception as e:
            return None, None, str(e)
    
    with tab4:
        st.subheader("Hitung Uji t Dua Sampel Dependen")
        st.markdown("Masukkan data sampel mentah.")
    
        st.subheader("Sampel 1")
        data1_input = st.text_area("Masukkan Data Sampel 1 (contoh: 12, 15, 10, 14, 13)", key="d1", height=100)
    
        st.subheader("Sampel 2")
        data2_input = st.text_area("Masukkan Data Sampel 2 (contoh: 18, 17, 19, 15, 20)", key="d2", height=100)
    
        st.markdown("---")
        alpha = st.slider("Tingkat Signifikansi (α)", 0.01, 0.10, 0.05, 0.01)
    
        if st.button("Hitung"):
            data1 = list_data(data1_input)
            x1, n1, err1 = parse_and_calculate_stats(data1)
            data2 = list_data(data2_input)
            x2, n2, err2 = parse_and_calculate_stats(data2)
            selisih = np.array(data2) - np.array(data1)
            dbar, n_d, err3 = parse_and_calculate_stats(selisih)
    
            if err1 or err2 or err3:
                st.error(err1 or err2 or err3)
            else:
                st.success("Statistik Sampel Berhasil Dihitung.")
    
                st.subheader("📌 Statistik Sampel")
                colA, colB, colC = st.columns(3)
                with colA:
                    st.write("**Sampel 1**")
                    st.write(f"$$n₁ = {n1}$$")
                    st.write(f"$$x̄₁ = {x1:.4f}$$")
                with colB:
                    st.write("**Sampel 2**")
                    st.write(f"$$n₂ = {n2}$$")
                    st.write(f"$$x̄₂ = {x2:.4f}$$")
                with colC:
                    st.write("**Selisih**")
                    st.write(f"$$n_{{d}} = {n_d}$$")
                    st.write(f"$$\\overline{{D}} = {dbar:.4f}$$")
    
                # ===========================
                # UJI t
                # ===========================
                S_D =  math.sqrt(np.sum((selisih - dbar)**2)/(n_d-1))
                SE = S_D/math.sqrt(n_d)
                t_value = dbar / SE
                t_tabel = t_tabel = t.ppf(1 - alpha/2, n_d - 1)
    
                st.subheader("📊 Hasil Uji t")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("t Hitung", f"{t_value:.4f}")
                with col2:
                    st.metric("t Tabel (dua sisi)", f"±{t_tabel:.4f}")
    
                st.subheader("🟩 Keputusan")
                if abs(t_value) > t_tabel:
                    st.error(f"TOLAK H₀ — Ada perbedaan rata-rata yang signifikan.")
                else:
                    st.info(f"GAGAL TOLAK H₀ — Tidak ada bukti perbedaan rata-rata.")
    
                st.markdown("---")
                st.subheader("Detail Tambahan")
                st.write(f"Standar Deviasi (SD) = {S_D:.4f}")
                st.write(f"Standard Error (SE) = {SE:.4f}")


