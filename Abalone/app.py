import pandas as pd
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Veriyi yükleme ve sütun isimlerini güncelleme
df = pd.read_csv('train.csv')
df.columns = df.columns.str.replace(r'[\s\.]', '_', regex=True)

# Bağımlı ve bağımsız değişkenlerin seçimi
x = df.drop(['id', 'Rings'], axis=1)
y = df[['Rings']]

# Eğitim ve test verilerini ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Ön işleme (StandardScaler ve OneHotEncoder)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1', 'Whole_weight_2', 'Shell_weight']),
        ('cat', OneHotEncoder(), ['Sex'])
    ]
)

# Streamlit uygulaması
def rings_pred(Sex, Length, Diameter, Height, Whole_weight, Whole_weight_1, Whole_weight_2, Shell_weight):
    input_data = pd.DataFrame({
        'Sex': [Sex],
        'Length': [Length],
        'Diameter': [Diameter],
        'Height': [Height],
        'Whole_weight': [Whole_weight],
        'Whole_weight_1': [Whole_weight_1],
        'Whole_weight_2': [Whole_weight_2],
        'Shell_weight': [Shell_weight]
    })
    
    
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('Abalone.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

st.title("Abalone Veri seti ile Yaş Tahmini Regresyon Modeli")
st.write("Veri Gir")

Sex = st.selectbox('Sex', df['Sex'].unique())
Length = st.selectbox('Length', df['Length'].unique())
Diameter = st.selectbox('Diameter', df['Diameter'].unique())
Height = st.selectbox('Height', df['Height'].unique())
Whole_weight = st.selectbox('Whole_weight', df['Whole_weight'].unique())
Whole_weight_1 = st.selectbox('Whole_weight_1', df['Whole_weight_1'].unique())
Whole_weight_2 = st.selectbox('Whole_weight_2', df['Whole_weight_2'].unique())
Shell_weight = st.selectbox('Shell_weight', df['Shell_weight'].unique())
    
if st.button('Predict'):
    rings = rings_pred(Sex, Length, Diameter, Height, Whole_weight, Whole_weight_1, Whole_weight_2, Shell_weight)
    st.write(f'The predicted rings is: {rings:.2f}')