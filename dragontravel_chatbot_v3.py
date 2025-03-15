import re
import datetime
import random
import json
import spacy.tokens
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
from langdetect import detect, DetectorFactory, detect_langs
import dateparser
import unicodedata
import locale

# Importar los módulos de reconocimiento de voz
import whisper
from IPython.display import Audio, display
from gtts import gTTS
from pysentimiento import create_analyzer

DetectorFactory.seed = 0 

# Simulated database for storage
bookings_db = []

# Base airport data - could be expanded in a real implementation
airports = {
    "IAH": {"name": "Houston", "code": "IAH"},
    "BER": {"name": "Berlin", "code": "BER"},
    "JFK": {"name": "New York", "code": "JFK"},
    "LHR": {"name": "London", "code": "LHR"},
    "CDG": {"name": "Paris", "code": "CDG"},
    "MEX": {"name": "Mexico City", "code": "MEX"},
    "MAD": {"name": "Madrid", "code": "MAD"},
    "BCN": {"name": "Barcelona", "code": "BCN"},
    "BOG": {"name": "Bogotá", "code": "BOG"},
    "LIM": {"name": "Lima", "code": "LIM"},
    "EZE": {"name": "Buenos Aires", "code": "EZE"},
    "SCL": {"name": "Santiago", "code": "SCL"},
    "DFW": {"name": "Dallas", "code": "DFW"},
    "MIA": {"name": "Miami", "code": "MIA"},
    "LAX": {"name": "Los Angeles", "code": "LAX"},
    "ORD": {"name": "Chicago O'Hare", "code": "ORD"},
    "CUN": {"name": "Cancún", "code": "CUN"},
    "YYZ": {"name": "Toronto Pearson", "code": "YYZ"},
    "SFO": {"name": "San Francisco", "code": "SFO"},
    "LAS": {"name": "Las Vegas", "code": "LAS"},
    "DEN": {"name": "Denver", "code": "DEN"},
    "PHX": {"name": "Phoenix", "code": "PHX"},
    "ATL": {"name": "Atlanta", "code": "ATL"},
    "MCO": {"name": "Orlando", "code": "MCO"},
    "EWR": {"name": "Newark", "code": "EWR"},
    "SEA": {"name": "Seattle-Tacoma", "code": "SEA"},
    "AUS": {"name": "Austin", "code": "AUS"},
    "BOS": {"name": "Boston Logan", "code": "BOS"},
    "SAN": {"name": "San Diego", "code": "SAN"},
    "PHL": {"name": "Philadelphia", "code": "PHL"},
    "CLT": {"name": "Charlotte Douglas", "code": "CLT"},
    "MSP": {"name": "Minneapolis-Saint Paul", "code": "MSP"},
    "DTW": {"name": "Detroit Metropolitan", "code": "DTW"},
    "IAD": {"name": "Washington Dulles", "code": "IAD"},
    "DCA": {"name": "Ronald Reagan Washington National", "code": "DCA"},
    "TPA": {"name": "Tampa", "code": "TPA"},
    "HOU": {"name": "Houston Hobby", "code": "HOU"},
    "BWI": {"name": "Baltimore/Washington", "code": "BWI"},
    "SJC": {"name": "San Jose", "code": "SJC"},
    "FLL": {"name": "Fort Lauderdale", "code": "FLL"},
    "RSW": {"name": "Southwest Florida", "code": "RSW"},
    "PIT": {"name": "Pittsburgh", "code": "PIT"},
    "RDU": {"name": "Raleigh-Durham", "code": "RDU"},
    "IND": {"name": "Indianapolis", "code": "IND"},
    "STL": {"name": "St. Louis", "code": "STL"},
    "MKE": {"name": "Milwaukee Mitchell", "code": "MKE"},
    "OKC": {"name": "Oklahoma City", "code": "OKC"},
    "OMA": {"name": "Omaha Eppley", "code": "OMA"},
    "GYE": {"name": "Guayaquil", "code": "GYE"},
    "UIO": {"name": "Quito", "code": "UIO"},
    "FCO": {"name": "Roma", "code": "FCO"},
    "LHR": {"name": "Londres", "code": "LHR"},
    "HND": {"name": "Tokyo", "code": "HND"},
}

travel_class_es = {
    "Economy": "Clase económica",
    "Business": "Clase ejecutiva",
    "First Class" : "Primera clase"
}

# Airlines for demonstration
airlines = ["DragonAir", "SkyWings", "GlobalFlyers", "AtlanticWay", "PacificRoute"]

class DragonTravelBot:
    def __init__(self):
        # Load NLP models
        print("Loading NLP models...")
        self.load_nlp_models()
        self.reset_booking()
        print("Bot initialized and ready!")
        
    def load_nlp_models(self):
        """Load the necessary NLP models for language understanding"""
        # Language identification model
        # self.lang_classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        
        # Load spaCy models for English and Spanish
        # This provides entity recognition, part-of-speech tagging, etc.
        spacy.prefer_gpu()
        try:
            self.nlp_en = spacy.load("en_core_web_md")
        except:
            # Fallback to small model if medium not available
            self.nlp_en = spacy.load("en_core_web_sm")
            
        try:
            self.nlp_es = spacy.load("es_core_news_md")
        except:
            # Fallback to small model if medium not available
            self.nlp_es = spacy.load("es_core_news_sm")
        

        
    def reset_booking(self, lang = None):                  #^ Añado lang para mantener coherencia linguistica en caso de reinicio
        """Reset the current booking information"""
        self.booking = {
            "num_passengers": None,
            "flight_type": None,
            "departure_airport": None,
            "arrival_airport": None,
            "departure_datetime": None,
            "arrival_datetime": None,
            "airline": None,
            "seat_class": None,
            "email": None
        }
        self.current_state = "greeting"
        self.language_set = False  # Flag para indicar si un lenguaje ha sido detectado
        if not lang:
            self.detected_language = "en"  # Default language
            self.nlp = self.nlp_en  # Default NLP pipeline
            self.responses = self.get_responses(self.detected_language)
        

    def detect_language(self, text):
        """Detect the language of the input text using langdetect"""
        try:
            lang_probs = detect_langs(text)
            lang_detected = max(lang_probs, key=lambda x: x.prob)  # Idioma con mayor probabilidad
            lang_code = lang_detected.lang
            confidence = lang_detected.prob
            
            # Si el idioma detectado es confiable (>70%), lo usamos
            if confidence > 0.7:
                if lang_code == "es":
                    self.detected_language = "es"
                    self.nlp = self.nlp_es
                elif lang_code == "en":
                    self.detected_language = "en"
                    self.nlp = self.nlp_en
                else:
                    # Si no es español ni inglés, asumimos inglés como fallback
                    self.detected_language = "en"
                    self.nlp = self.nlp_en

            # Manejo de Spanglish: si detecta inglés y español con valores similares
            elif "en" in [l.lang for l in lang_probs] and "es" in [l.lang for l in lang_probs]:
                self.detected_language = "es" if "es" in [l.lang for l in lang_probs if l.prob > 0.4] else "en"
                self.nlp = self.nlp_es if self.detected_language == "es" else self.nlp_en

            else:
                self.detected_language = "en"  # Fallback
                self.nlp = self.nlp_en

            self.responses = self.get_responses(self.detected_language)
            return self.detected_language

        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"  # Default a inglés en caso de error

    
    def set_language(self, language):
        self.detected_language = language
        self.nlp = self.nlp_es if language == "es" else self.nlp_en
        self.responses = self.get_responses(language)

        if language == 'es':
            locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")
        else:
            locale.setlocale(locale.LC_TIME, "en_US.UTF-8")

    
    def process_message(self, message):
        """Process an incoming message and return a response"""
        if not message.strip():
            return self.responses["empty_message"]
        
        # Detectar idioma
        if not self.language_set:
            detected_lang = self.detect_language(message)
            self.set_language(detected_lang)
            self.language_set = True
            
        
        #^ Añadir Salir
        if message.strip().lower() == "exit" or message.strip().lower() == "salir":
            self.reset_booking(lang=self.detected_language)
            return self.responses["quit"]
            
        # comandos para cambiar el idioma
        if message.strip().lower() == "switch to english" or message.strip().lower() == "english please":
            self.detected_language = "en"
            self.nlp = self.nlp_en
            self.responses = self.get_responses("en")
            print("Switching to English. How can I help you with your travel plans?")
            
        if message.strip().lower() == "cambiar a español" or message.strip().lower() == "español por favor":
            self.detected_language = "es"
            self.nlp = self.nlp_es
            self.responses = self.get_responses("es")
            print("Cambiando a español. ¿Cómo puedo ayudarte con tus planes de viaje?")
        
        # Process the message with the NLP pipeline
        doc = self.nlp(message)
        
        # In the greeting state, try to extract flight info from the initial message
        #* 1. "greeting"
        previous = None
        if self.current_state == "greeting":
            print(self.responses["welcome"])
            # Extract any flight information if provided
            extracted_info = self.extract_flight_info(doc, message)

            if extracted_info.get("departure"):
                self.booking["departure_airport"] = extracted_info["departure"]
                self.current_state = "collect_arrival"
            
            if extracted_info.get("destination"):
                self.booking["arrival_airport"] = extracted_info["destination"]

            if extracted_info.get("date"):
                self.booking["departure_datetime"] = extracted_info["date"]
            
            if extracted_info.get("flight_type"):
                self.booking["flight_type"] = extracted_info["flight_type"]

            # Establecer current state            
            previous = self.determine_next_state()
            # print(f"DEBUG: {extracted_info = }")
        
        # Handle other conversation states
        response = self.handle_conversation_state(message, doc)
        if previous:
            response = str(previous) + "\n" + str(response)
            previous = None

        return response

    def handle_conversation_state(self, message, doc):
        """Handle the conversation based on the current state"""
        self.responses = self.get_responses(self.detected_language)

        # print(f"DEBUG: {self.current_state =}")

        #* 2. "collect_departure"
        if self.current_state == "collect_departure":
            airport = self.extract_airport(doc, message)
            if airport:
                self.booking["departure_airport"] = airport
                self.current_state = "collect_arrival"
                response = self.responses["departure_collected"].format(airport=self.get_airport_name(airport))
                return response
            else:
                response = self.responses["departure_not_understood"]
                return response

        #* 3. "collect_arrival"        
        elif self.current_state == "collect_arrival":
            airport = self.extract_airport(doc, message)
            if airport:
                self.booking["arrival_airport"] = airport
                self.current_state = "collect_date"
                response = self.responses["arrival_collected"].format(
                  dep_airport=self.get_airport_name(self.booking["departure_airport"]),
                  arr_airport=self.get_airport_name(airport)
                )
                return response
            else:
                response = self.responses["arrival_not_understood"]
                return response
        
        #* 4. "collect_date"
        elif self.current_state == "collect_date":
            if self.booking["departure_datetime"]:
                self.current_state = "collect_trip_type"
                response = self.responses["date_collected_only"].format(
                    date=self.booking["departure_datetime"].strftime(self.responses["date_format"])
                )
                return response

            date = self.extract_date(doc, message)
            if date:
                self.booking["departure_datetime"] = date
                self.current_state = "collect_trip_type"
                response = self.responses["date_collected_only"].format(
                    date=self.booking["departure_datetime"].strftime(self.responses["date_format"])
                )
                return response
            else:
                response = self.responses["date_not_understood"]
                return response

        #* 5. "collect_trip_type"
        elif self.current_state == "collect_trip_type":
            flight_type = self.extract_flight_type(message)
            if flight_type:
                self.booking["flight_type"] = flight_type
                if flight_type == "round_trip":
                    self.current_state = "collect_return_date"
                    response = self.responses["round_trip_selected"]
                    return response
                else:
                    if self.booking["num_passengers"] is not None:
                        self.current_state = "collect_seat_class"
                        response = self.responses["passengers_collected"].format(passengers=self.booking["num_passengers"])
                        return response
                    
                    self.current_state = "collect_passengers"
                    response = self.responses["one_way_selected"]
                    return response
            else:
                response = self.responses["trip_type_not_understood"]
                return response
        
        #* 6. "collect_return_date"
        elif self.current_state == "collect_return_date":
            date = self.extract_date(doc, message)
            if date:
                self.booking["arrival_datetime"] = date
                self.current_state = "collect_passengers"
                response = self.responses["return_date_collected"].format(
                    date=date.strftime(self.responses["date_format"])
                )
                return response
            else:
                response = self.responses["return_date_not_understood"]
                return response
        
        #* 7. "collect_passangers"
        elif self.current_state == "collect_passengers":
            if self.booking["num_passengers"] is not None:
                self.current_state = "collect_seat_class"
                response = self.responses["passengers_collected"].format(passengers=self.booking["num_passengers"])
                return response

            passengers = self.extract_number(message)
            if passengers:
                self.booking["num_passengers"] = passengers
                self.current_state = "collect_seat_class"
                response = self.responses["passengers_collected"].format(passengers=passengers)
                return response
            else:
                response = self.responses["passengers_not_understood"]
                return response

        #* 8. "collect_seat class"
        elif self.current_state == "collect_seat_class":
            seat_class = self.extract_seat_class(message)
            if seat_class:
                self.booking["seat_class"] = seat_class
                self.current_state = "collect_email"
                response = self.responses["seat_class_collected"].format(
                    seat_class=travel_class_es[seat_class] if self.detected_language == "es" else seat_class
                    )
                return response
            else:
                response = self.responses["seat_class_not_understood"]
                return response
        
        #* 9. "collect_email"
        elif self.current_state == "collect_email":
            email = self.extract_email(message)
            if email:
                self.booking["email"] = email
                # Assign a random airline for this prototype
                self.booking["airline"] = random.choice(airlines)
                self.current_state = "confirm_details"
                response = self.get_confirmation_message()
                return response
            else:
                response = self.responses["email_not_understood"]
                return response
        
        #* 10. "confirm_details"
        elif self.current_state == "confirm_details":
            confirmation = self.extract_confirmation(message)
            # print(f"DEBUG: {confirmation = }")
            if confirmation == "yes":
                self.initial_language = self.detected_language
                booking_id = self.save_booking()
                self.last_booking_id = booking_id 
                self.generate_quotation()
                self.send_email_quotation()
                response = self.responses["booking_confirmed"].format(booking_id=booking_id)
                self.reset_booking(lang=self.detected_language)

                self.current_state = "ask_feedback"
                return response + "\n" + self.responses["feedback_prompt"]
            elif confirmation == "no":
                self.current_state = "greeting"
                self.reset_booking(lang=self.detected_language)
                return self.responses["booking_restart"]
            else:
                return self.responses["confirmation_not_understood"]
        
        #* 11. "ask_feedback"
        elif self.current_state == "ask_feedback":
            self.set_language(self.initial_language)
            confirmation = self.extract_confirmation(message)
            # self.responses = self.get_responses(self.initial_language)
            if confirmation == "yes":
                self.current_state = "getting_feedback"
                return self.responses["feedback_ready"]
            else:
                self.current_state = "greeting"
                self.reset_booking(lang=self.detected_language)
                return self.responses["feedback_skipped"]
        
        #* 12. "getting_feedback"
        elif self.current_state == "getting_feedback":
            audio_path = "feedback.mp3"  # Ruta de ejemplo
            return self.process_feedback(audio_path)
        
        else:
            # Handle unexpected states
            #^ Envío el parámetro del idioma para mantener coherencia
            self.reset_booking(lang=self.detected_language)
            return self.responses["error_restart"]
    
    
    def get_airport_name(self, code):
      """Obtiene el nombre del aeropuerto basado en su código."""
      return airports.get(code, {}).get("name", code)

    
    def extract_flight_info(self, doc: spacy.tokens.doc.Doc, message):
        """Extract flight information from an initial query using NLP"""
        result = {
            "departure": None,
            "destination": None,
            "date": None,
            "flight_type": None
        }
        
        # Try to find city names that might be airports
        potential_cities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

        if self.detected_language == "en":
            # Look for "from X to Y" pattern in English
            from_to_match = re.search(r'from\s+(\w+(?:\s+\w+)*)\s+to\s+(\w+(?:\s+\w+)*)', message.lower())
        
        elif self.detected_language == "es":
            # Look for "de X a Y" pattern in Spanish
            from_to_match = re.search(r'de\s+(\w+(?:\s+\w+)*)\s+a\s+(\w+(?:\s+\w+)*)', message.lower())
        
        if from_to_match:
            departure_text = from_to_match.group(1)
            destination_text = from_to_match.group(2)
            
            result["departure"] = self.text_to_airport_code(departure_text)
            result["destination"] = self.text_to_airport_code(destination_text)
        elif len(potential_cities) >= 2:
            # If we found at least two cities, assume the first is departure and second is arrival
            # This is a simplification for the prototype
            result["departure"] = self.text_to_airport_code(potential_cities[0])
            result["destination"] = self.text_to_airport_code(potential_cities[1])
        
        # Extract date
        result["date"] = self.extract_date(doc, message)

        # Try to extract flight type
        result["flight_type"] = self.extract_flight_type(message)
    
        # passengers = self.extract_number(message)
        # if passengers:
        #     result["passengers"] = passengers
            
        return result
    
    def determine_next_state(self):
        """
        Determine the next state of conversation based on what information 
        we already have and what we still need to collect
        """
        missing_info = []

        summary_actions = {
            "collect_return_date" : {"en":"Return date:", "es":"Fecha de retorno:"},
            "collect_trip_type" : {"en":"Trip type:", "es":"Tipo de vuelo:"},
            "collect_date" : {"en":"Departure date:", "es":"Fecha de salida:"},
            "collect_arrival" : {"en":"Arrival:", "es":"Llegada:"},
            "collect_departure" : {"en":"Departure:", "es":"Salida:"},
        }

        #  Asumiendo que recolectamos toda la información, el siguiente estado:
        self.current_state = "collect_passangers"

        # Verificar si "collect_return_date" procede
        if self.booking["flight_type"] == "round_trip" and not self.booking["arrival_datetime"]:
            self.current_state = "collect_return_date"
        elif self.booking["flight_type"] == "round_trip" and self.booking["arrival_datetime"]:
            missing_info.append(f"{summary_actions["collect_return_date"][self.detected_language]} {self.booking['arrival_datetime'].strftime(self.responses['date_format'])}")

        # Verificar si "collect_trip_type" procede
        if not self.booking["flight_type"]:
            self.current_state = "collect_trip_type"
        else:
            missing_info.append(f"{summary_actions["collect_trip_type"][self.detected_language]} {self.booking["flight_type"]}")
        
        # Verificar si "collect_date" procede
        if not self.booking["departure_datetime"]:
            self.current_state = "collect_date"
        else:
            missing_info.append(f"{summary_actions["collect_date"][self.detected_language]} {self.booking['departure_datetime'].strftime(self.responses['date_format'])}")
        
        # Verificar si "collect_arrival" procede
        if not self.booking["arrival_airport"]:
            self.current_state = "collect_arrival"
        else:
            missing_info.append(f"{summary_actions["collect_arrival"][self.detected_language]} {self.get_airport_name(self.booking['arrival_airport'])}")
        
        # Verificar si "collect_departure" procede
        if not self.booking["departure_airport"]:
            self.current_state = "collect_departure"
        else:
            missing_info.append(f"{summary_actions["collect_departure"][self.detected_language]} {self.get_airport_name(self.booking['departure_airport'])}")


        if self.booking["seat_class"]:
            missing_info.append(f"{self.booking['seat_class'] if self.detected_language == "en" else travel_class_es[self.booking['seat_class']]}")       
        
        # Create a summary of what we've understood so far
        # if len(missing_info) > 2:
        #     missing_info[2] = missing_info[2] + "\n"
        summary = ', '.join(missing_info[::-1])
        
        if self.detected_language == "es":
            return str(f"Entiendo que buscas un vuelo con {summary}.")
        else:
            return str(f"I understand you're looking for a flight with {summary}.")
        
    
    def remove_accents(self, text):
      """Delete accents."""
      return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


    def text_to_airport_code(self, city_text):
        """Convert a city name text to an airport code"""
        city_text = self.remove_accents(city_text.lower().strip())
        
        # Check if it matches a known airport code directly
        upper_text = city_text.upper()
        if upper_text in airports:
            return upper_text
            
        # Check if it matches or partially matches a city name
        for code, airport_info in airports.items():
            airport_name = self.remove_accents(airport_info["name"].lower())
            if city_text == airport_name or city_text in airport_name or airport_name in city_text:
                return code
                
        return None


    def extract_airport(self, doc: spacy.tokens.doc.Doc, message):
        """Extract airport from text using NLP"""
        # First look for location entities
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                airport_code = self.text_to_airport_code(ent.text)
                if airport_code:
                    return airport_code
        
        # Fallback to simple text matching
        for word in message.split():
            airport_code = self.text_to_airport_code(word)
            if airport_code:
                return airport_code
                
        return None
    

    def extract_date(self, doc: spacy.tokens.doc.Doc, message):
        """Extraer fecha de un texto usando NLP-spaCy"""
        # First try to extract DATE entities using spaCy
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # Convert the date entity to a datetime object
                try:
                    return self.parse_date_string(ent.text)
                except:
                    pass
        
        # Fallback to regex pattern matching for common date formats
        return self.extract_date_with_regex(message)


    def parse_date_string(self, date_text):
      """Convert a date string to a datetime object with better day handling."""
      today = datetime.datetime.now()
      date_text = date_text.lower().strip()

      # Manejo de fechas relativas
      if "tomorrow" in date_text or "mañana" in date_text:
          return today + datetime.timedelta(days=1)
      elif "next week" in date_text or "próxima semana" in date_text:
          return today + datetime.timedelta(weeks=1)

      # Diccionario de meses en inglés y español
      months = {
          **{
              "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
              "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
              "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, 
              "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
          },
          **{
              "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
              "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
              "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, 
              "ago": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dic": 12
          }
      }

      # Expresión regular mejorada para capturar "8 de junio" y similares
      date_match = re.search(r'(\d{1,2})\s*(?:de\s+)?([a-zA-Z]+)(?:\s+de\s+(\d{4}))?', date_text)

      if date_match:
          day = int(date_match.group(1))  # Día capturado correctamente
          month_name = date_match.group(2)  # Nombre del mes
          year = int(date_match.group(3)) if date_match.group(3) else today.year  # Año opcional

          # Buscar el número del mes en el diccionario
          month = months.get(month_name.lower())
          if month:
              return datetime.datetime(year, month, day, 12, 0)  # Asigna correctamente el día

      # Usar dateparser
      parsed_date = dateparser.parse(date_text, languages=["es", "en"])
      if parsed_date:
          return parsed_date

      raise ValueError("Could not extract a valid date from input")


    def extract_date_with_regex(self, text):
        """Extract date using regex patterns for both MM/DD/YYYY and '8 de junio' formats"""
        text = text.lower().strip()

        # Detectar formato MM/DD/YYYY o DD/MM/YYYY
        date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', text)
        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            year = int(date_match.group(3))
            if year < 100:
                year += 2000  # Ajuste de año corto (ej. 23 → 2023)
            return datetime.datetime(year, month, day, 12, 0)

        # date_match = re.search(r'(\d{1,2})\s*(?:al\s+(\d{1,2})\s+de\s+)?([a-zA-Z]+)(?:\s+de\s+(\d{4}))?', text)
        date_match = re.search(r'(\d{1,2})\s*(?:al\s+(\d{1,2})\s+de\s+|de\s+)([a-zA-Z]+)(?:\s+de\s+(\d{4}))?', text)
        # print(f"DEBUG: {date_match = }")
        if date_match:
            if self.current_state == "collect_return_date" and date_match.group(2) is not None:
                day = int(date_match.group(2))  # Extrae segunda fecha si existe
                month_name = date_match.group(3)  # Nombre del mes
                year = int(date_match.group(4)) if date_match.group(4) else datetime.datetime.now().year  # Año opcional
            else:
                day = int(date_match.group(1))  # Extrae  días de un solo dígito
                month_name = date_match.group(3)  # asume el mismo mes
                year = int(date_match.group(4)) if date_match.group(4) else datetime.datetime.now().year  # Año opcional

            # Diccionario de meses en español e inglés
            months = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, 
                "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
                "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
                "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
                "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, 
                "ago": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dic": 12
            }

            month = months.get(month_name.lower())  # Convierte el mes a número
            if month:
                return datetime.datetime(year, month, day, 12, 0)

        # Usar dateparser, si regex no funciona
        parsed_date = dateparser.parse(text, languages=["es", "en"])
        if parsed_date:
            return parsed_date

        return None 
    
    def extract_flight_type(self, message):
      """Extract flight type (one-way or round-trip)"""
      message = message.lower()

      if self.detected_language == "es":
        if "ida y vuelta" in message or "redondo" in message or "regreso" in message:
            return "round_trip"
        if "ida" in message or "solo ida" in message or "solo de ida" in message:
            return "one_way"

      # Patterns for one-way in both English and Spanish
      one_way_patterns = [
          "one way", "one-way", "oneway", "single", "sencillo", "solamente ida"
      ]
      
      # Patterns for round-trip in both English and Spanish
      round_trip_patterns = [
          "round trip", "round-trip", "roundtrip", "return"
      ]
      
      # Check for one-way patterns
      for pattern in round_trip_patterns:
        if re.search(r"\b" + re.escape(pattern) + r"\b", message):
            return "round_trip"
      
      # Check for one-way patterns
      for pattern in one_way_patterns:
        if re.search(r"\b" + re.escape(pattern) + r"\b", message):
            return "one_way"
    
      return None
    
    def extract_number(self, message):
      """Extract a number from text."""
      message = message.lower().strip()

      # Buscar números explícitos (dígitos)
      match = re.search(r'\b(\d+)\b', message)
      if match:
          num = int(match.group(1))
          return num

      # Buscar números en palabras pero evitando falsos positivos como "personas"
      number_words = {
          "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
          "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
          "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
      }

      words = message.split()
      for i, word in enumerate(words):
          if word in number_words:
              # Evitar falsos positivos como "personas"
              if i + 1 < len(words) and words[i + 1] in ["persona", "personas", "pasajero", "pasajeros", "people", "person"]:
                  num = number_words[word]
                  return num

      return None

    
    def extract_seat_class(self, message):
        """Extract seat class preference"""
        message = message.lower()
        
        # English and Spanish seat class patterns
        if any(term in message for term in ["economy", "coach", "economica", "económica","economico" , "turista"]):
            return "Economy"
        elif any(term in message for term in ["business", "ejecutiva", "ejecutivo" , "negocios"]):
            return "Business"
        elif any(term in message for term in ["first", "primera"]):
            return "First Class"
            
        return None
    
    def extract_email(self, message):
        """Extract email address from message"""
        match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', message)
        if match:
            return match.group(0)
        return None
    
    def extract_confirmation(self, message):
        """Extract confirmation (yes/no) from message"""
        message = message.lower().strip('.')
        
        # Yes patterns in English and Spanish
        yes_patterns = ["yes", "yeah", "yep", "correct", "right", "ok", "okay", "sure",
                        "si", "sí", "claro", "correcto", "de acuerdo", "vale"]
        
        # No patterns in English and Spanish
        no_patterns = ["no", "nope", "wrong", "incorrect", "not", "nah",
                      "no es correcto", "incorrecto", "mal"]
        
        for pattern in yes_patterns:
            if pattern in message.split():
                return "yes"
                
        for pattern in no_patterns:
            if pattern in message.split():
                return "no"
                
        return None
    
    def get_confirmation_message(self):
        """Generate a confirmation message with all booking details"""
        return self.responses["confirmation_message"].format(
            dep_airport=self.get_airport_name(self.booking["departure_airport"]),
            arr_airport=self.get_airport_name(self.booking["arrival_airport"]),
            dep_date=self.booking['departure_datetime'].strftime(self.responses["date_format"]),
            ret_date=self.booking['arrival_datetime'].strftime(self.responses["date_format"]) if self.booking['flight_type'] == "round_trip" and self.booking['arrival_datetime'] else "N/A",
            trip_type=self.responses["round_trip" if self.booking['flight_type'] == "round_trip" else "one_way"],
            passengers=self.booking['num_passengers'],
            seat_class=self.booking['seat_class'],
            airline=self.booking['airline'],
            email=self.booking['email']
        )
    
    def save_booking(self):
        """Save booking to database (simulated)"""
        # Generate a simple booking ID
        booking_id = len(bookings_db) + 1001
        self.booking["booking_id"] = booking_id
        
        # Add a timestamp
        self.booking["created_at"] = datetime.datetime.now()
        
        # Add to our simulated database
        bookings_db.append(self.booking.copy())
        
        return booking_id
    
    def generate_quotation(self):
        """Generate a price quotation based on booking details"""
        # Simple pricing model for prototype
        base_prices = {
            "Economy": 500,
            "Business": 1500,
            "First Class": 3000
        }
        
        # Base price depends on class
        base_price = base_prices.get(self.booking["seat_class"], 500)
        
        # Adjust for flight type
        if self.booking["flight_type"] == "round_trip":
            base_price *= 1.8  # 10% discount on return leg
            
        # Multiply by passengers
        total = base_price * self.booking["num_passengers"]
        
        # Add random variance (10%)
        variance = random.uniform(0.9, 1.1)
        total *= variance
        
        # Round to nearest 10
        total = round(total / 10) * 10
        
        # Save to booking
        self.booking["quotation_amount"] = total
        
        return total
    
    def send_email_quotation(self):
        """Simulate sending an email with the quotation"""
        # In a real implementation, this would connect to an SMTP server
        quotation = self.booking["quotation_amount"]
        print(f"[EMAIL SIMULATION] Quotation of ${quotation:.2f} sent to {self.booking['email']}")
        
        # Return success for prototype
        return True
    def process_feedback(self, audio_path):
        """Procesa un archivo de audio existente para extraer feedback del cliente"""
        try:
            model = whisper.load_model("medium") 
            result = model.transcribe(audio_path)
            transcription = result["text"]
            
            category = self.categorize_feedback(transcription)
            self.save_feedback(transcription, category)

            self.current_state = "greeting"
            self.reset_booking(lang=self.detected_language)
            
            return self.responses["feedback_received"].format(category=category)
        except Exception as e:
            print(f"Error processing audio: {e}")
            return self.responses["feedback_error"]

    def categorize_feedback(self, text):
        """Clasifica el feedback como queja o felicitación usando pysentimiento"""
        analyzer = create_analyzer(task="sentiment", lang=self.initial_language)
        sentiment = analyzer.predict(text).output
        if sentiment == "NEG":
          return "Queja"
        elif sentiment == "POS":
          return "Felicitación"
        elif sentiment == "NEU":
          return "Neutral"
        else:
            return "Otro"
    
    def save_feedback(self, transcription, category):
        """Guarda el feedback en la base de datos simulada"""
        feedback_entry = {
            "booking_id": self.last_booking_id,
            "audio_transcription": transcription,
            "language": self.initial_language,
            "category": category,
            "timestamp": datetime.datetime.now()
        }
        bookings_db.append(feedback_entry)
    
    def get_responses(self, language):
        """Get response templates for the specified language"""
        if language == "es":
            return {
                "empty_message": "No he recibido ningún mensaje. ¿En qué puedo ayudarte?",
                "welcome": "¡Bienvenido a DragonTravel! ¿Desde dónde te gustaría volar?",
                "departure_collected": "¡Excelente! Vuelo desde {airport}. ¿Cuál es tu destino?",
                "departure_not_understood": "No pude entender ese aeropuerto. ¿Podrías especificar tu ciudad de salida o código de aeropuerto?",
                "arrival_collected": "Vuelo desde {dep_airport} a {arr_airport}. ¿Cuándo te gustaría salir?",
                "arrival_not_understood": "No pude entender ese aeropuerto. ¿Podrías especificar tu ciudad de destino o código de aeropuerto?",
                "airports_collected": "Veo que quieres volar de {dep_airport} a {arr_airport}. ¿Cuándo te gustaría salir?",
                "date_collected": "Veo que quieres volar de {dep_airport} a {arr_airport} el {dep_date}. ¿Es un vuelo de ida solamente o de ida y vuelta?",
                "date_collected_only": "Saliendo el {date}. ¿Es un vuelo de ida solamente o de ida y vuelta?",
                "date_not_understood": "No pude entender esa fecha. Por favor especifica una fecha de salida (por ejemplo, 15 de octubre de 2025).",
                "round_trip_selected": "¿Cuándo te gustaría regresar?",
                "one_way_selected": "¿Cuántos pasajeros viajarán?",
                "trip_type_not_understood": "¿Es un vuelo de ida solamente o de ida y vuelta?",
                "return_date_collected": "Regresando el {date}. ¿Cuántos pasajeros viajarán?",
                "return_date_not_understood": "No pude entender esa fecha. Por favor especifica una fecha de regreso (por ejemplo, 25 de octubre de 2025).",
                "passengers_collected": "{passengers} pasajero(s). ¿Qué clase de asiento prefieres? (Económica, Ejecutiva o Primera Clase)",
                "passengers_not_understood": "¿Cuántos pasajeros viajarán? Por favor proporciona un número.",
                "seat_class_collected": "{seat_class} seleccionada. Por favor proporciona tu dirección de correo electrónico para la cotización.",
                "seat_class_not_understood": "Por favor selecciona clase Económica, Ejecutiva o Primera Clase.",
                "email_not_understood": "No pude reconocer esa dirección de correo electrónico. Por favor proporciona un email válido.",
                "confirmation_message": "Por favor confirma los detalles de tu reserva:\n\n" +
                                       "Desde: {dep_airport}\n" +
                                       "A: {arr_airport}\n" +
                                       "Salida: {dep_date}\n" +
                                       "Regreso: {ret_date}\n" +
                                       "Tipo de vuelo: {trip_type}\n" +
                                       "Pasajeros: {passengers}\n" +
                                       "Clase: {seat_class}\n" +
                                       "Aerolínea: {airline}\n" +
                                       "Email: {email}\n\n" +
                                       "¿Es correcta esta información?",
                "confirmation_not_understood": "¿Es correcta esta información? Por favor confirma con sí o no.",
                "booking_confirmed": "¡Excelente! Tu reserva (ID: {booking_id}) ha sido confirmada. Se ha enviado una cotización a tu correo electrónico.",
                "booking_restart": "Empecemos de nuevo. ¿Desde dónde te gustaría volar?",
                "error_restart": "Lo siento, algo salió mal. Empecemos de nuevo. ¿Desde dónde te gustaría volar?",
                "date_format": "%d de %B, %Y",
                "round_trip": "Ida y vuelta",
                "one_way": "Solo ida",
                "feedback_prompt": "¿Quieres dejar algún comentario?",
                "feedback_skipped": "¡Gracias por elegir DragonTravel!",
                "feedback_received": "Gracias, recibimos tu comentario",
                "feedback_not_understood": "No pude reconocer el audio",
                "feedback_error": "Lo siento, algo salió mal al procesar tu audio",
                "feedback_ready": "Déjanos saber que piensas.",
                "quit": "Chao"
            }
        else:  # English and fallback
            return {
                "empty_message": "I didn't receive any message. How can I help you?",
                "welcome": "Welcome to DragonTravel! Where would you like to fly from?",
                "departure_collected": "Great! Flying from {airport}. What's your destination?",
                "departure_not_understood": "I couldn't understand that airport. Could you please specify your departure city or airport code?",
                "arrival_collected": "Flying from {dep_airport} to {arr_airport}. When would you like to depart?",
                "arrival_not_understood": "I couldn't understand that airport. Could you please specify your destination city or airport code?",
                "airports_collected": "I see you want to fly from {dep_airport} to {arr_airport}. When would you like to depart?",
                "date_collected": "I see you want to fly from {dep_airport} to {arr_airport} on {dep_date}. Is this a one-way or round-trip flight?",
                "date_collected_only": "Departing on {date}. Is this a one-way or round-trip flight?",
                "date_not_understood": "I couldn't understand that date. Please specify a departure date (e.g., October 15, 2025).",
                "round_trip_selected": "When would you like to return?",
                "one_way_selected": "How many passengers will be traveling?",
                "trip_type_not_understood": "Is this a one-way or round-trip flight?",
                "return_date_collected": "Returning on {date}. How many passengers will be traveling?",
                "return_date_not_understood": "I couldn't understand that date. Please specify a return date (e.g., October 25, 2025).",
                "passengers_collected": "{passengers} passenger(s). What seat class would you prefer? (Economy, Business, or First Class)",
                "passengers_not_understood": "How many passengers will be traveling? Please provide a number.",
                "seat_class_collected": "{seat_class} class selected. Please provide your email address for the quotation.",
                "seat_class_not_understood": "Please select Economy, Business, or First Class.",
                "email_not_understood": "I couldn't recognize that email address. Please provide a valid email.",
                "confirmation_message": "Please confirm your booking details:\n\n" +
                                      "From: {dep_airport}\n" +
                                      "To: {arr_airport}\n" +
                                      "Departure: {dep_date}\n" +
                                      "Return: {ret_date}\n" +
                                      "Flight type: {trip_type}\n" +
                                      "Passengers: {passengers}\n" +
                                      "Class: {seat_class}\n" +
                                      "Airline: {airline}\n" +
                                      "Email: {email}\n\n" +
                                      "Is this information correct?",
                "confirmation_not_understood": "Is this information correct? Please confirm with yes or no.",
                "booking_confirmed": "Great! Your booking (ID: {booking_id}) has been confirmed. A quotation has been sent to your email.",
                "booking_restart": "Let's start over. Where would you like to fly from?",
                "error_restart": "I'm sorry, something went wrong. Let's start over. Where would you like to fly from?",
                "date_format": "%B %d, %Y",
                "round_trip": "Round-trip",
                "one_way": "One-way",
                "feedback_prompt": "Do you want give feedback?",
                "feedback_skipped": "Thank you for choosing DragonTravel!",
                "feedback_received": "Thanks, we receive your feedback",
                "feedback_not_understood": "I couldn't understand that audio.",
                "feedback_error": "I'm sorry, something went wrong.",
                "feedback_ready": "Let us know what you think.",
                "quit": "Bye!"

            }