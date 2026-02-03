# app.py - MINIMAL WORKING VERSION
import matplotlib
matplotlib.use('Agg')  # ← THIS IS KEY
import matplotlib.pyplot as plt
import streamlit as st

st.title("SMS Spam Classifier")
st.write("Matplotlib test")

# Create a simple plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Test Plot")
st.pyplot(fig)

st.success("✅ App is working!")
