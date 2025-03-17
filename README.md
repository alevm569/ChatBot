Chatbot DragonTravel
====================

Descripción General
-------------------

DragonTravel es un chatbot multilingüe habilitado por voz para una agencia de viajes ficticia que ayuda a los usuarios a reservar boletos de avión. El chatbot puede procesar consultas en inglés, español o spanglish, creando una experiencia de reserva fluida en varios idiomas.

Los usuarios pueden interactuar con el bot para buscar vuelos, reservar boletos y recibir cotizaciones de precios. El sistema guía a los usuarios a través de una conversación natural para recopilar toda la información necesaria para reservar un vuelo.
Características

---------------

* 🌐 **Soporte Multilingüe**: Maneja conversaciones en inglés, español e incluso lenguaje mixto (spanglish)
* 🗣️ **Interacción por Voz**: Procesa entradas de voz y responde con salida de voz
* 🔍 **Comprensión del Lenguaje Natural**: Extrae detalles de vuelos del lenguaje conversacional
* ✈️ **Reserva de Vuelos**: Recopila toda la información necesaria para reservas de vuelos
* 💰 **Generación de Cotizaciones**: Crea cotizaciones de precios basadas en detalles de reserva
* 📧 **Integración de Correo Electrónico**: Simula el envío de cotizaciones por correo electrónico
* 📝 **Recopilación de Comentarios**: Recopila y categoriza comentarios de usuarios

Tecnologías Utilizadas
----------------------

* **Python** - Lenguaje de programación principal
* **spaCy** - Para procesamiento de lenguaje natural (NLP)
* **Whisper** - Para conversión de voz a texto
* **gTTS** (Google Text-to-Speech) - Para conversión de texto a voz
* **langdetect** - Para detección de idioma
* **dateparser** - Para entender fechas en diferentes formatos e idiomas
* **pandas** - Para manejo de datos
* **pysentimiento** - Para análisis de sentimiento de comentarios

Estructura del Proyecto
-----------------------

    dragontravel/
    ├── dragontravel_chatbot_v3.py  # Implementación principal del chatbot
    ├── prototype_sound.ipynb       # Notebook Jupyter con la función para manejo de audio
    ├── prototype_test.ipynb        # Notebook Jupyter con ejemplos de interacción
    ├── requirements.txt            # dependencias para la ejecución
    └── README.md                   # Este archivo

Instalación
-----------

1. Clona este repositorio:
      git clone https://github.com/tuusuario/dragontravel-chatbot.git
      cd dragontravel-chatbot

2. Crea un entorno virtual (recomendado):
      python -m venv venv
      source venv/bin/activate  # En Windows: venv\Scripts\activate

3. Instala las dependencias requeridas:
      pip install -r requirements.txt

4. Descarga los modelos de spaCy necesarios:
      python -m spacy download en_core_web_md
      python -m spacy download es_core_news_md
   
   

Uso
---

### Interfaz de Línea de Comandos

Ejecuta el chatbot con interacción por voz:

    

    from sound_wrap import audio_chat_interaction
    # Inicia la interfaz de chat por voz
    audio_chat_interaction()

### Uso Programático

    from dragontravel_chatbot_v3 import DragonTravelBot
    
    # Inicializa el chatbot
    bot = DragonTravelBot()
    
    # Procesa un mensaje
    respuesta = bot.process_message("Necesito un vuelo de Houston a Berlín en octubre")
    print(respuesta)

### Notebook Jupyter

Explora el notebook `prototype_test.ipynb`  y `prototype_sound.ipynb`para ejemplos interactivos de cómo usar el chatbot.
Ejemplos de Conversaciones

--------------------------

### Español

    Usuario: "Necesito un vuelo de Guayaquil a París en noviembre"
    Bot: "¡Excelente! Vuelo desde Guayaquil. ¿Cuál es tu destino?"
    Usuario: "París"
    Bot: "Vuelo desde Guayaquil a París. ¿Cuándo te gustaría salir?"
    Usuario: "15 de noviembre"
    Bot: "Saliendo el 15 de noviembre, 2025. ¿Es un vuelo de ida solamente o de ida y vuelta?"
    ...

### Inglés

    Usuario: "I need a flight from Houston to Berlin in October"
    Bot: "Great! Flying from Houston. What's your destination?"
    Usuario: "Berlin"
    Bot: "Flying from Houston to Berlin. When would you like to depart?"
    Usuario: "October 15th"
    Bot: "Departing on October 15, 2025. Is this a one-way or round-trip flight?"
    ...

Requisitos
----------

* Python 3.8+
* Ver `requirements.txt` para todas las dependencias de paquetes

Mejoras Futuras
---------------

* Integración con APIs reales de vuelos para reservas actuales
* Soporte para más idiomas
* Capacidades conversacionales mejoradas
* Interfaz web para interacción de chat
* Integración con aplicaciones móviles

Licencia
--------

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
Aviso Legal

-----------

DragonTravel es una agencia de viajes ficticia creada con fines de demostración. Este chatbot no reserva vuelos reales ni procesa pagos.
