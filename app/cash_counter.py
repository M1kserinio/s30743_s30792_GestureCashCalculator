class CashCounter:

    def __init__(self):
        self.all_objects = []  # Wszystkie wykryte banknoty z YOLO
        self.saved_group = None  # Zapisany nominał
        self.current_r_filter = None  # Aktualny gest prawej ręki
        self.last_operation_msg = "Gotowy do operacji"

    def set_objects(self, objects):
        self.all_objects = objects

    def set_r_filter(self, filter_name):
        """Ustawia filtr z prawej dłoni"""
        self.current_r_filter = filter_name

    def save_group(self):
        """L_OK: Zapisuje grupę lub resetuje zapis"""
        if self.current_r_filter is not None and self.current_r_filter != "ALL":
            self.saved_group = self.current_r_filter
            self.last_operation_msg = f"ZAPISANO: {self.saved_group} zl"
        else:
            self.saved_group = None
            self.last_operation_msg = "ZRESETOWANO ZAPIS"

    def calculate_add(self):
        """L_POINT: Dodawanie grup lub zliczanie sztuk"""
        # Gdy brak gestu R, ten sam nominał lub brak zapisu to zliczamy sztuki zamiast kwotyt
        if self.current_r_filter is None or self.current_r_filter == self.saved_group or self.saved_group is None:
            target = self.saved_group or "WSZYSTKO"

            count = 0
            total = 0.0
            for obj in self.all_objects:
                if self._match(obj, target):
                    count += 1
                    total += obj["value"]

            self.last_operation_msg = f"LICZNIK [{target}]: {count} szt. ({total:.2f} zl)"

        # Gdy mamy dwie różne grupy (Zapisana + Bieżąca) to dodajemy ich kwoty
        else:
            count = 0
            total = 0.0
            for obj in self.all_objects:
                label = str(obj["label"])
                # Jeśli banknot należy do zapisanej lub bieżącej grupy
                if str(self.saved_group) in label or str(self.current_r_filter) in label:
                    count += 1
                    total += obj["value"]

            self.last_operation_msg = f"DODANO ({self.saved_group}zl + {self.current_r_filter}zl): {total:.2f} zl ({count} szt.)"

    def calculate_subtract(self):
        """L_TWO: Odejmowanie bieżącej grupy lub zliczanie bez wybranej grupy."""
        # Gdy brak gestu R lub ten sam nominał to liczymy wszystko BEZ wskazanej grupy
        if self.current_r_filter is None or self.current_r_filter == self.saved_group or self.saved_group is None:
            exclude = self.current_r_filter or self.saved_group

            count = 0
            total = 0.0
            for obj in self.all_objects:
                label = str(obj["label"])
                # Jeśli banknot ma wykluczany nominał, pomijamy go
                if exclude is not None and exclude != "ALL" and str(exclude) in label:
                    continue
                count += 1
                total += obj["value"]

            if exclude is not None and exclude != "ALL":
                self.last_operation_msg = f"BEZ [{exclude} zl]: {count} szt. ({total:.2f} zl)"
            else:
                self.last_operation_msg = f"LACZNIE: {count} szt. ({total:.2f} zl)"

        # Gdy mamy dwie różne grupy to odejmujemy: Zapisana - Bieżąca
        else:
            val_saved = 0.0
            val_current = 0.0
            for obj in self.all_objects:
                label = str(obj["label"])
                if str(self.saved_group) in label:
                    val_saved += obj["value"]
                if str(self.current_r_filter) in label:
                    val_current += obj["value"]

            diff = val_saved - val_current
            self.last_operation_msg = f"ROZNICA ({self.saved_group}zl - {self.current_r_filter}zl): {diff:.2f} zl"

    def get_visible_objects(self):
        """Pokazuje tylko banknoty pasujace do zapisanej grupy lub biezacego gestu."""
        # Jesli R_OK lub brak jakichkolwiek filtrow to pokazujemy wszystko
        if self.current_r_filter == "ALL":
            return self.all_objects
        if self.saved_group is None and self.current_r_filter is None:
            return self.all_objects

        visible = []
        for obj in self.all_objects:
            pasuje_zapisana = self._match(obj, self.saved_group) if self.saved_group else False
            pasuje_biezaca = self._match(obj, self.current_r_filter) if (self.current_r_filter and self.current_r_filter != "ALL") else False

            if pasuje_zapisana or pasuje_biezaca:
                visible.append(obj)

        return visible

    def _match(self, obj, group_name):
        """Czt etykieta banknotu pasuje do filtra"""
        if group_name == "WSZYSTKO" or group_name == "ALL" or group_name is None:
            return True

        try:
            return int(obj["value"]) == int(group_name)
        except (ValueError, TypeError):
            return False


    def total_value(self):
        """Liczy łączną wartość wszystkich banknotów na stole."""
        total = 0.0
        for obj in self.all_objects:
            total += obj["value"]
        return total