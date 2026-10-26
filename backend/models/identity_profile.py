from pydantic import BaseModel
from typing import Optional


class IdentityProfile(BaseModel):

    name: str|None=None
    first_name: str|None=None
    middle_name: str|None=None
    last_name: str|None=None

    phone: str |None=None
    email: str |None=None

    dob: str|None=None

    address: str|None=None
    house_number: str|None=None
    street: str|None=None
    locality: str|None=None
    city: str|None=None
    state: str|None=None
    pincode: str|None=None

    guardian_name: str|None=None
    place_of_birth: str|None=None

    gender: str|None=None
    year_of_birth: str|None=None

    aadhaar_number: str|None=None
    pan_number: str|None=None
    voter_id: str|None=None
    birth_registration_number: str|None=None
    passport_number: str|None=None