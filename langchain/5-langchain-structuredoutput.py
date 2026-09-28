#  Structured output means asking the LLM to return data in a fixed format/schema instead of free-form text. like JSON.
# Pydantic lets you define the expected structure using a Python class and validates the model's output.
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
# it gives strcuted, validated at run time and nested structure
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    job: str = Field(description="Person's job")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# structured_model = model.with_structured_output(Person)

# response = structured_model.invoke(
#     "Eshwar is a 25 year old software developer."
# )

# print(response)


# Nested Strucuture
class Actor(BaseModel):
    name: str
    role: str


class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(
        None,
        description="Budget in ruppee"
    )


# model_with_structure = model.with_structured_output(MovieDetails)

# response = model_with_structure.invoke(
#     "Provide details about the movie Avengers"
# )

# print(response)



#==========================================
#TypeDict
# No runtime validation but provides Nested output
from typing_extensions import TypedDict

class Person(TypedDict):
    name: str
    age: int
    job: str

# structured_model = model.with_structured_output(Person)

# response = structured_model.invoke(
#     "Eshwar is 27 years old and works as a software developer."
# )
# print(structured_model)
# print(response)
# {'name': 'Eshwar', 'age': 27, 'job': 'software developer'}

"""
TypedDict mainly gives type/schema information to Python and LangChain. It does not provide the runtime validation features that Pydantic provides.

TypedDict → lightweight structured dictionary
Pydantic  → structured object + runtime validation
"""

#===========================================================
# DataClasses
"""
Pydantic
→ Schema + validation
→ Excellent for LLM structured output
→ Field descriptions
→ Type validation
→ Nested models

TypedDict
→ Type/schema hints
→ Returns dictionary
→ Lightweight
→ No Pydantic-style runtime validation

Dataclass
→ Python data container
→ Type hints
→ Lightweight
→ No built-in validation like Pydantic
"""

from dataclasses import dataclass, fields
from typing import Optional, get_type_hints
from pydantic import create_model


@dataclass
class Actor:
    name: str
    role: str


@dataclass
class MovieDetails:
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: Optional[float] = None


def dataclass_to_pydantic(dc):
    """Convert a dataclass to a Pydantic model for structured output."""
    field_defs = {}
    for f in fields(dc):
        field_defs[f.name] = (get_type_hints(dc).get(f.name, str), ...)
    return create_model(dc.__name__, **field_defs)


MovieDetailsModel = dataclass_to_pydantic(MovieDetails)
model_with_structure = model.with_structured_output(MovieDetailsModel)

response = model_with_structure.invoke(
    "Provide details about the movie Avengers"
)

print(">>DataClasses",response)