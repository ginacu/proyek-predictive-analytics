# Laporan Proyek Machine Learning - Gina Cahya Utami

## Domain Proyek
Pada proyek ini, saya memilih proyek domain Ekonomi dan Bisnis. Dalam dunia bisnis, para pebisnis akan berupaya keras untuk mencari tahu bagaimana cara meningkatkan keuntungan bisnis mereka. Begitu pula dengan bisnis penjualan rumah. Dalam praktiknya pun pastilah terdapat tantangan serta hambatan di tengah persaingan, meskipun demikian, pebisnis harus bisa memutar otak agar tantangan dan hambatan tersebut bukanlah menjadi penghambat dalam berbisnis.

Salah satu tantangan yang dihadapi dalam bisnis penjualan yakni kondisi lingkungan. Kondisi lingkungan di sini yakni seperti maraknya kasus kriminal, kekerasan, perampokan, dan lain sebagainya. Namun apakah dengan kondisi lingkungan tersebut dapat memberikan sebuah kesempatan untuk pebisnis dalam menjalankan bisnisnya, atau malah sebaliknya?

Dengan menggunakan dataset dari [tautan](https://www.kaggle.com/sandeep04201988/housing-price-index-using-crime-rate-data), saya akan memprediksi apakah kondisi lingkungan yang mempunyai riwayat kriminalitas akan membantu dalam penjualan rumah, atau bahkan sebaliknya? Dalam memprediksi permasalahan tersebut saya menggunakan 3 algoritma, yakni Linear Regression, KNN, dan Random Forest.

## Business Understanding
Sebuah perusahaan penjualan rumah akan menjual rumah di daerah dengan kondisi lingkungan yang terdapat kasus kriminal, perusahaan tersebut tentu berpikir apakah rumah yang kita jual tersebut akan laku di pasaran? Dengan sedikitnya jumlah kriminal maka diharapkan akan semakin meningkat harga jual rumahnya.

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi harga jual rumah untuk menjawab permasalahan berikut :
- Dari serangkaian fitur atau kondisi kriminal yang ada, fitur apa yang paling berpengaruh terhadap harga jual rumah?
- Berapa harga jual rumah jika mempunyai fitur atau kondisi kriminal tertentu?

### Goals
Untuk  menjawab pertanyaan tersebut, kita akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur yang paling berkorelasi dengan harga jual rumah.
- Membuat model machine learning yang dapat memprediksi harga jual rumah seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
Untuk menyelesaikan permasalahan di atas, kami menggunakan tiga algoritma machine learning sebagai solusinya, yaitu Linear Regression, K-Nearest Neighbor, dan Random Forest.
- **Linear Regression**. Pemilihan metode regresi linear sebagai metode prediksi pada penelitian ini didasari oleh kelebihannya dalam menaksir parameter model yang sederhana dan data yang berbasis runtun waktu. Selain itu, metode ini dapat melakukan analisis dengan menggunakan beberapa variabel bebas (X) sehingga hasil prediksi bisa lebih akurat. Kekurangannya adalah karena hasil prediksi dari linear regression merupakan nilai estimasi, sehingga kemungkinan untuk tidak sesuai dengan data aktual tetaplah ada. Cara kerja Linear Regression yakni memanggil fungsi LinearRegression() yang kita import dari library scikit-learn.
- **K-Nearest Neighbor**. Pemilihan metode K-Nearest Neighbor sebagai metode prediksi pada penelitian ini didasari oleh kelebihannya yang mudah dipahami dan diimplementasikan, tangguh terhadap data training sample yang noisy, dan memiliki konsistensi yang kuat. Kekurangannya yakni perlu menentukan parameter k (jumlah tetangga terdekat), sensitif terhadap data outlier. Cara kerja K-Nearest Neighbor yakni dengan memanggil fungsi KNeighborsRegressor() yang kita import dari library scikit-learn.
- **Random Forest**. Pemilihan metode Random Forest sebagai metode prediksi pada penelitian ini didasari oleh kelebihannya yaitu dapat mengatasi noise dan missing value serta dapat mengatasi data dalam jumlah yang besar. Dan kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data. Cara kerja Random Forest yakni dengan memanggil fungsi RandomForestRegressor() yang kita import dari library scikit-learn.

## Data Understanding
Data yang saya gunakan yakni diunduh dari situs [Kaggle](https://www.kaggle.com/sandeep04201988/housing-price-index-using-crime-rate-data). Data tersebuh berisi indeks harga nsa atau non seasonal index yang merupakan fitur yang akan saya prediksi, kemudia fitur yang lain merupakan kondisi kriminal di sekitar rumah tersebuh yang akan menjadi pertimbangan dalam prediksi ini.
Deskripsi variabel pada dataset adalah sebagai berikut:
- Year = Tahun
- index_nsa = Harga rumah (non seasonal index)
- City, State = Kota, Negara Bagian
- Population = Populasi
- Violent Crimes = Kekerasan
- Homicides = Pembunuhan
- Rapes = Perkosaan
- Assaults = Penyerangan
- Robberies = Perampokan

Dalam tahap ini saya melakukan Data loading dan proses EDA yang saya representasikan menggunakan visualisasi, diantaranya :
Exploratory Data Analysis - Menangani Missing Value dan Outliers, visualisasinya menggunakan boxplot dari library seaborn
Exploratory Data Analysis - Univariate Analysis, visualisasinya menggunakan plot dan histogram dari library matplotlib
Exploratory Data Analysis - Multivariate Analysis, visualisasinya menggunakan catplot dari library seaborn

## Data Preparation
Teknik yang saya gunakan pada tahapan Data Preparation adalah :
- Train-Test-split, teknik ini berguna untuk membagi data menjadi data uji dan data latih. Teknik ini menggunakan fungsi train_test_split dari library scikit-learn.
- Standarisasi, teknik ini membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Di sini kita mengurangkan mean (nilai rata-rata) pada seluruh fitur numerik kemudian membaginya dengan standar deviasi untuk menggeser distribusi menggunakan fungsi StandardScaler() dari library scikit-learn. Setelah kita mengecek informasi menggunakan fungsi .describe(), kita mengetahui bahwa mean pada fitur numerik berubah menjadi 0 dan standar deviasi-nya menjadi 1.

## Modeling
Pada tahap ini, saya mengembangkan model machine learning dengan tiga algoritma, yakni Linear Regression, K-Nearest Neighbor, Random Forest. Langkah selanjutnya yakni mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik.
Langkah pertama dalam proses modeling yakni menyiapkan sebuah DataFrame baru untuk menampung berapa nilai mae-nya yang berfungsi pada proses analisis model. Proses pembuatannya menggunakan kode berikut :
```
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['LinearRegression', 'KNN', 'RandomForest'])
```

Langkah kedua yakni kita akan membandingkan ketiga algoritma seperti yang sudah kita jelaskan di atas, berikut prosesnya :
- **Linear Regression**. Berikut kode program pada proses modeling menggunakan Linear Regression :
    ```sh
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    models.loc['train_mse','LinearRegression'] = mean_squared_error(y_pred=lin_reg.predict(x_train), y_true=y_train)
    ```
    Langkah pertama kita import terlebih dahulu fungsi LinearRegression() pada library scikit-learn. Setelah itu kita definisikan fungsi LinearRegression() ke dalam variabel baru bernama lin_reg. Kemudian kita uji model tersebut data uji kita yakni x_train dan y_train menggunakan fungsi .fit().
    
    Setelah kita uji model terebut, kita cek akurasinya menggunakan metrik mae dan masukan nilai mae nya ke DataFrame yang telah kita buat sebelumnya.
    
- **K-Nearest Neighbor**. Berikut kode program pada proses modeling menggunakan K-Nearest Neighbor :
    ```sh
    from sklearn.neighbors import KNeighborsRegressor
    
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_train)
    
    models.loc['train_mse','KNN'] = mean_squared_error(y_pred=knn.predict(x_train), y_true=y_train)
    ```
    Langkah pertama kita import terlebih dahulu fungsi KNeighborsRegressor() pada library scikit-learn. Setelah itu kita definisikan fungsi KNeighborsRegressor() ke dalam variabel baru bernama knn. Kita masukkan juga parameter n_neighbors nya yakni berjumlah 10, yang berarti kita mendefinisikan nilai tetangga-nya 10. Kemudian kita uji model tersebut pada data uji kita yakni x_train dan y_train menggunakan fungsi .fit().
    
    Setelah kita uji model terebut, kita cek akurasinya menggunakan metrik mae dan masukan nilai mae nya ke DataFrame yang telah kita buat sebelumnya.
    
- **Random Forest**. Berikut kode program pada proses modeling menggunakan Random Forest :
    ```sh
    from sklearn.ensemble import RandomForestRegressor
    
    RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
    RF.fit(x_train, y_train)
     
    models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)
    ```
    Langkah pertama kita import terlebih dahulu fungsi RandomForestRegressor() pada library scikit-learn. Setelah itu kita definisikan fungsi RandomForestRegressor() ke dalam variabel baru bernama RF. Pada fungsi tersebut kita tambahkan parameter n_estimators-nya 50, max_depth-nya 16, random_state-nya 55, n_jobs-nya -1. Kemudian kita uji model tersebut data uji kita yakni x_train dan y_train menggunakan fungsi .fit().
    
    Setelah kita uji model terebut, kita cek akurasinya menggunakan metrik mae dan masukan nilai mae nya ke DataFrame yang telah kita buat sebelumnya.
## Evaluation
Metrik evaluasi yang digunakan untuk mengukur kinerja model di atas yakni menggunakan Mean Squared Error (MSE) dan cek hasil akurasi.
- **Mean Squared Error (MSE)**. Teknik ini menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. Berikut kode programnya :
    ```
    mse = pd.DataFrame(columns=['train', 'test'], index=['LinearRegression','KNN', 'RandomForest'])
    model_dict = {'LinearRegression': lin_reg, 'KNN': knn, 'RandomForest':RF}
    for name, model in model_dict.items():
        mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3 
        mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3
    
    mse
    ```
    setelah di jalankan, outputnya yakni :
    |  | train | test |
    | ------ | ------ | ------ |
    | LinearRegression | 0.892576 | 0.96617 |
    | KNN | 0.380629 | 0.52106 |
    | RandomForest | 0.0454171 | 0.334078 |
    Dapat dilihat kalau Random Forest mempunyai eror paling sedikit dibanding LinearRegression dan RandomForest.

- **Accuracy**. Mengukur kinerja model menggunakan fungsi .score() dalam skala 100. Berikut kode programnya :
    ```
    print("Accuracy score dari model Linear Regression = ", lin_reg.score(x_test, y_test))
    print("Accuracy score dari model KNN               = ", knn.score(x_test, y_test))
    print("Accuracy score dari model Random Forest     = ", RF.score(x_test, y_test))
    ```
    Dan output dari kode program di atas yakni :
    Accuracy score dari model Linear Regression =  0.760807233952006
    Accuracy score dari model KNN               =  0.8710022235399519
    Accuracy score dari model Random Forest     =  0.9172929820665391

Dapat dilihat akurasi pada model Random Forest yang paling tinggi yakni mencapai lebih dari 90%.

### Kesimpulan
Setelah melakukan analisis di atas, dapat kita jawab Problem Statements dan Goal yang saya buat di atas, yakni :
- Fitur-fitur kriminalitas di atas mempunyai korelasi yang kecil terhadap harga, sehingga tidak terlalu memengaruhi harga jual rumah.
- Model machine learning yang akurat sesuai dengan metrik evaluasi dari MSE dan accuracy yakni menggunakan model Random Forest.

**---Ini adalah bagian akhir laporan---**