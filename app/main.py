"""Cámara, material y destino. Ejecutar con streamlit run app/main.py."""
import hashlib
import html
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import streamlit as st

from app.detector import analyze_objects,annotate_objects
from app.localization import manual_region
from app.recycling import guidance
from app.resources import get_detector,get_model,reserved_image_hashes
from app.browser_camera import BrowserRecognition,CAMERA_GUIDE,camera_component
from app.settings import checkpoint_signature,load_settings
from app.vision import decode_image,annotate_region


def show_result(reading,threshold):
    info=guidance(reading.material)
    uncertain=reading.confidence<threshold
    title=('Posible ' if uncertain else '')+info['label'].lower()
    color='green' if reading.material=='organicos' else 'white'
    st.markdown(f'<div class="waste-result"><h3>{html.escape(title.capitalize())}</h3>'
                f'<p class="destination"><span class="bin-dot {color}"></span>'
                f'{html.escape(info["destination"])}</p></div>',unsafe_allow_html=True)
    st.caption(info['advice'])
    if uncertain:st.caption('No está claro el material. Acércalo y mejora la luz.')


def show_objects(objects,threshold):
    if not objects:st.info('Muestra un residuo completo frente a la cámara.')
    readings={}
    for item in objects:
        reading=item.reading
        previous=readings.get(reading.material)
        if previous is None or reading.confidence>previous.confidence:readings[reading.material]=reading
    for reading in readings.values():show_result(reading,threshold)


def live_panel(service,detector,threshold,model_key):
    previous=st.session_state.get('browser_recognition')
    if previous is None or previous[0]!=model_key:
        previous=(model_key,BrowserRecognition(threshold));st.session_state['browser_recognition']=previous
    recognition=previous[1]
    def receive_frame():
        recognition.consume(st.session_state.get('browser_camera'),detector,service)
    payload=camera_component(response=recognition.response,guide=CAMERA_GUIDE,interval_ms=250,key='browser_camera',
                             on_change=receive_frame,default=None)
    recognition.consume(payload,detector,service)


def photo_panel(mode,service,detector,settings,model_key):
    upload=(st.camera_input('Tomar una fotografía',key='camera_photo') if mode=='Tomar foto' else
            st.file_uploader('Selecciona una imagen',type=['jpg','jpeg','png','webp'],key='upload_photo'))
    if upload is None:return
    try:
        payload=upload.getvalue()
        rgb=decode_image(payload,settings['max_upload_mb']*1024*1024,settings['max_image_pixels'],reserved_image_hashes())
        key=(hashlib.sha256(payload).hexdigest(),model_key)
        previous=st.session_state.get('photo_result')
        if previous is None or previous[0]!=key:
            with st.spinner('Reconociendo residuo…'):objects=analyze_objects(rgb,detector,service)
            st.session_state['photo_result']=(key,objects)
        else:objects=previous[1]
        picture,result=st.columns([1.8,1],gap='large')
        with picture:st.image(annotate_objects(rgb,objects),width='stretch')
        with result:
            if objects:show_objects(objects,settings['low_confidence_threshold'])
            else:st.info('No se reconoció un residuo. Prueba otra foto con más luz y el objeto completo.')
        with st.expander('Ajustar el recorte si no lo reconoce'):
            horizontal=st.slider('Horizontal (%)',0,100,(20,80),key='manual_x')
            vertical=st.slider('Vertical (%)',0,100,(20,80),key='manual_y')
            bounds=(horizontal[0]/100,vertical[0]/100,horizontal[1]/100,vertical[1]/100)
            region=manual_region(rgb,bounds)
            st.image(annotate_region(rgb,region),width=400)
            if st.button('Reconocer recorte'):
                show_result(service.classify(rgb,region=region),settings['low_confidence_threshold'])
    except Exception as exc:st.error(f'No se pudo reconocer la imagen: {exc}')


def main():
    st.set_page_config(page_title='Vision-IA',page_icon='♻',layout='wide',initial_sidebar_state='collapsed')
    st.markdown(f'<style>{(ROOT/"app/style.css").read_text(encoding="utf-8")}</style>',unsafe_allow_html=True)
    st.title('Vision-IA')
    st.caption('Muestra un residuo a la vez. Identifica el material y dónde va.')
    settings=load_settings()
    mode=st.radio('Origen de la imagen',['Cámara en vivo','Tomar foto','Subir imagen'],horizontal=True,key='image_source',label_visibility='collapsed')
    try:
        model_key=(*checkpoint_signature(settings['checkpoint']),settings['cpu_threads'])
        candidate=settings.get('material_candidate_checkpoint')
        refiner=settings.get('material_refiner_checkpoint')
        if candidate or refiner:
            signature = checkpoint_signature(candidate) if candidate else (None, None, None)
            model_key+=(*signature, settings.get('material_candidate_threshold',.4),settings.get('material_candidate_weight'))
        if refiner:
            model_key+=(*checkpoint_signature(refiner), settings.get('material_refiner_weight'),
                        tuple(settings.get('material_refiner_preserve', ())),
                        tuple(settings.get('material_refiner_preserve_consensus', ())))
            calibration=settings.get('material_refiner_calibration')
            if calibration:
                model_key+=checkpoint_signature(calibration)
        service=get_model(*model_key)
        detector_key=(*checkpoint_signature(settings['detector_checkpoint']),settings['cpu_threads'],
                      settings['detection_threshold'],settings.get('detection_nms_iou',.45),
                      tuple(map(tuple,settings.get('detection_views',()))))
        detector=get_detector(*detector_key)
    except Exception as exc:
        st.error(f'No se pudo cargar el reconocimiento: {exc}');return
    if mode=='Cámara en vivo':
        live_panel(service,detector,settings['low_confidence_threshold'],(model_key,detector_key))
    else:
        st.session_state.pop('browser_recognition',None)
        photo_panel(mode,service,detector,settings,(model_key,detector_key))


if __name__=='__main__':main()
