# Aplicación Vision-IA

Interfaz Streamlit local y Cloud para cámara en vivo y fotografías. Los checkpoints de VisionLocaliza-D3, las CNN y el Transformer propio se configuran en `config/app.json`.

La cámara se elige en el navegador y requiere HTTPS o localhost. La caja sigue al residuo entre respuestas del detector. Los pesos del detector y de ambas CNN se incluyen en `models/runtime/`.

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1
```

Abre http://127.0.0.1:8501. Uso, despliegue y límites actuales: [README](../README.md).
