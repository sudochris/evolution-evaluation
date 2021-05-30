class HeaderValueMismatch(Exception):
    def __init__(self, header_length, value_length):
        self.message = f"{header_length=} != {value_length=}"
        super().__init__(self.message)


class InvalidHeader(Exception):
    pass
