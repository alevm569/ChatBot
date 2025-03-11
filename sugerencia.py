def extract_initial_booking_info(self, doc, message):
    """
    Extract all possible booking information from the initial query
    This helps create a more natural conversation by acknowledging 
    all the information the user has already provided
    """
    extracted_info = {
        "departure_airport": None,
        "arrival_airport": None,
        "departure_date": None,
        "return_date": None,
        "num_passengers": None,
        "flight_type": None,
        "seat_class": None
    }
    
    # Extract airports using both pattern matching and NER
    from_to_info = self.extract_from_to_airports(message)
    if from_to_info.get("departure"):
        extracted_info["departure_airport"] = from_to_info.get("departure")
    if from_to_info.get("destination"):
        extracted_info["arrival_airport"] = from_to_info.get("destination")
    
    # Try to extract dates - might be departure only or both departure and return
    dates = self.extract_dates(doc, message)
    if dates and len(dates) >= 1:
        extracted_info["departure_date"] = dates[0]
        # If two dates are mentioned, assume second is return date
        if len(dates) >= 2:
            extracted_info["return_date"] = dates[1]
            # Having both departure and return dates implies round trip
            extracted_info["flight_type"] = "round_trip"
    
    # Look for explicit flight type indicators
    if any(word in message.lower() for word in ["round", "round-trip", "return", "ida y vuelta", "redondo"]):
        extracted_info["flight_type"] = "round_trip"
    elif any(word in message.lower() for word in ["one way", "one-way", "single", "ida", "sencillo"]):
        extracted_info["flight_type"] = "one_way"
    
    # Try to extract number of passengers
    num_passengers = self.extract_number_of_passengers(message)
    if num_passengers:
        extracted_info["num_passengers"] = num_passengers
    
    # Try to extract seat class
    seat_class = self.extract_seat_class(message)
    if seat_class:
        extracted_info["seat_class"] = seat_class
    
    return extracted_info

def extract_from_to_airports(self, message):
    """Extract departure and destination airports using pattern matching"""
    result = {"departure": None, "destination": None}
    
    # English patterns
    from_to_match = re.search(r'from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+?)(?:\s+in|\s+on|,|$)', message.lower())
    
    # Spanish patterns
    if not from_to_match:
        from_to_match = re.search(r'de\s+([A-Za-z\s]+?)\s+a\s+([A-Za-z\s]+?)(?:\s+en|\s+el|,|$)', message.lower())
    
    # Extract and convert to airport codes
    if from_to_match:
        departure_text = from_to_match.group(1).strip()
        destination_text = from_to_match.group(2).strip()
        
        result["departure"] = self.text_to_airport_code(departure_text)
        result["destination"] = self.text_to_airport_code(destination_text)
    
    return result

def extract_dates(self, doc, message):
    """Extract all dates mentioned in the message"""
    dates = []
    
    # Use spaCy's NER to find DATE entities
    date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    
    # For each date entity, try to convert it to a datetime object
    for date_text in date_entities:
        try:
            date_obj = self.parse_date_string(date_text)
            if date_obj:
                dates.append(date_obj)
        except:
            pass
    
    # Also look for month names if no specific dates were found
    if not dates:
        months = ["january", "february", "march", "april", "may", "june", "july", 
                 "august", "september", "october", "november", "december",
                 "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", 
                 "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        
        # Match month patterns
        for month in months:
            # English: "in March" or Spanish: "en marzo"
            if f" in {month}" in message.lower() or f" en {month}" in message.lower():
                # Create a date for the middle of the month
                current_year = datetime.datetime.now().year
                month_index = months.index(month) % 12 + 1  # Get month number (1-12)
                dates.append(datetime.datetime(current_year, month_index, 15, 12, 0))
                break
    
    return dates

def extract_number_of_passengers(self, message):
    """Extract number of passengers if mentioned"""
    # Look for patterns like "2 passengers", "for 3 people", etc.
    passengers_match = re.search(r'(\d+)\s+(passenger|passengers|people|person|adults|adult|personas|adultos)', message.lower())
    if passengers_match:
        return int(passengers_match.group(1))
    return None

def process_message(self, message):
    """Process an incoming message and return a response"""
    if not message.strip():
        return self.responses["empty_message"]
    
    # Detect language on first interaction
    if not self.language_set:
        self.detect_initial_language(message)
        self.language_set = True
    
    # Process with NLP pipeline
    doc = self.nlp(message)
    
    # For the first interaction, try to extract all possible booking information
    if self.current_state == "greeting":
        initial_info = self.extract_initial_booking_info(doc, message)
        
        # Update booking with extracted information
        if initial_info["departure_airport"]:
            self.booking["departure_airport"] = initial_info["departure_airport"]
        
        if initial_info["arrival_airport"]:
            self.booking["arrival_airport"] = initial_info["arrival_airport"]
        
        if initial_info["departure_date"]:
            self.booking["departure_datetime"] = initial_info["departure_date"]
        
        if initial_info["return_date"]:
            self.booking["arrival_datetime"] = initial_info["return_date"]
        
        if initial_info["flight_type"]:
            self.booking["flight_type"] = initial_info["flight_type"]
        
        if initial_info["num_passengers"]:
            self.booking["num_passengers"] = initial_info["num_passengers"]
        
        if initial_info["seat_class"]:
            self.booking["seat_class"] = initial_info["seat_class"]
        
        # Determine next state based on what information we already have
        return self.determine_next_state()
    
    # Handle other conversation states
    return self.handle_conversation_state(message, doc)

def determine_next_state(self):
    """
    Determine the next state of conversation based on what information 
    we already have and what we still need to collect
    """
    missing_info = []
    
    # Check what information is still missing
    if not self.booking["departure_airport"]:
        self.current_state = "collect_departure"
        return self.responses["welcome"]
    else:
        missing_info.append(f"departure: {self.booking['departure_airport']}")
    
    if not self.booking["arrival_airport"]:
        self.current_state = "collect_arrival"
        return self.responses["departure_collected"].format(airport=self.booking["departure_airport"])
    else:
        missing_info.append(f"arrival: {self.booking['arrival_airport']}")
    
    if not self.booking["departure_datetime"]:
        self.current_state = "collect_date"
        return self.responses["arrival_collected"].format(
            dep_airport=self.booking["departure_airport"],
            arr_airport=self.booking["arrival_airport"]
        )
    else:
        missing_info.append(f"departure date: {self.booking['departure_datetime'].strftime(self.responses['date_format'])}")
    
    if not self.booking["flight_type"]:
        self.current_state = "collect_trip_type"
        return self.responses["date_collected"].format(
            dep_airport=self.booking["departure_airport"],
            arr_airport=self.booking["arrival_airport"],
            dep_date=self.booking["departure_datetime"].strftime(self.responses["date_format"])
        )
    
    if self.booking["flight_type"] == "round_trip" and not self.booking["arrival_datetime"]:
        self.current_state = "collect_return_date"
        return self.responses["round_trip_selected"]
    elif self.booking["flight_type"] == "round_trip" and self.booking["arrival_datetime"]:
        missing_info.append(f"return date: {self.booking['arrival_datetime'].strftime(self.responses['date_format'])}")
    
    if not self.booking["num_passengers"]:
        self.current_state = "collect_passengers"
        if self.booking["flight_type"] == "round_trip":
            return self.responses["return_date_collected"].format(
                date=self.booking["arrival_datetime"].strftime(self.responses["date_format"])
            )
        else:
            return self.responses["one_way_selected"]
    else:
        missing_info.append(f"passengers: {self.booking['num_passengers']}")
    
    if not self.booking["seat_class"]:
        self.current_state = "collect_seat_class"
        return self.responses["passengers_collected"].format(passengers=self.booking["num_passengers"])
    else:
        missing_info.append(f"class: {self.booking['seat_class']}")
    
    # If we have almost everything, go to email collection
    self.current_state = "collect_email"
    
    # Create a summary of what we've understood so far
    summary = ', '.join(missing_info)
    
    if self.detected_language == "es":
        return f"Entiendo que buscas un vuelo con {summary}. Por favor proporciona tu dirección de correo electrónico para la cotización."
    else:
        return f"I understand you're looking for a flight with {summary}. Please provide your email address for the quotation."