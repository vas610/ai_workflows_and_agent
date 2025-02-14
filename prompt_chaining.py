#!.venv/bin/python3
# Prompt Chaining Example
import logging
import os
from datetime import datetime
from typing import Optional
import random

from ollama import ChatResponse, chat
from pydantic import BaseModel, Field

# -----------------------------------------------------#
# Set up logging configuration                         #
# -----------------------------------------------------#

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------#
# Define variables                                     #
# -----------------------------------------------------#
MODEL_NAME = "llama3.1"

# -----------------------------------------------------#
# Define the data models for each stage using Pydantic #
# -----------------------------------------------------#


class CheckTicketBooking(BaseModel):
    """Check if this request is about booking a new ticket"""

    description: str = Field(description="Raw description of the user input")
    is_ticket_booking: bool = Field(
        description="Whether this text describes booking a new flight or airline ticket"
    )


class ExtractTicketInfo(BaseModel):
    """Check if this request is about booking a new ticket"""

    departure: str = Field(description="Departure location of the flight"),
    destination: str = Field(
        description="Destination location of the flight"
    ),
    departure_date: Optional[datetime] = Field(
        default_factory=lambda: None, description="Departure date of the flight"),
    return_date: Optional[datetime] = Field(
        default_factory=lambda: None, description="Date when trip ends (if applicable else None)"),


class GenerateConfirmation(BaseModel):
    confirmation_message: str = Field(
        "A confirmation message for the ticket booking that includes the departure, destination, dates and ticket ID")

# -----------------------------------------------------#
# Define the functions for making the calls to LLMs    #
# -----------------------------------------------------#


def check_ticket_booking(user_input: str) -> CheckTicketBooking:
    """First LLM call to determine if input is a new flight or airline ticket booking request"""
    logger.info("Checking if the input describes booking a flight new ticket")
    logger.debug(f"Input text: {user_input}")

    response = chat(model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Analyze if the given text requests for booking a flight ticket.",
                        },
                        {"role": "user", "content": user_input},
                    ],
                    options={"temperature": 0.0},
                    # -- Structured output using the given data model
                    format=CheckTicketBooking.model_json_schema(),
                    )
    result = CheckTicketBooking.model_validate_json(response.message.content)
    logger.info(
        f"Flight ticket booking check complete - Is ticket booking: {
            result.is_ticket_booking}"
    )
    return result


def extract_ticket_info(user_input: str) -> CheckTicketBooking:
    """Second LLM call to extract ticket information like source, destination, and dates"""
    logger.info("Start extracting ticket information")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    today_date_for_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    response = chat(model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": f"{today_date_for_context}.\
                                Extract ticket information like source, destination, dates and return  as JSON .\
                If no source is mentioned, then say departure is missing.\
                If no destination is mentioned, then set it as None. Do not set random date.\
                If no return date is not mentioned, then set it as None.\
                ",
                        },
                        {"role": "user", "content": user_input},
                    ],
                    options={"temperature": 0.0},
                    # --> Structured output using the given data model
                    format=ExtractTicketInfo.model_json_schema(),
                    )
    result = ExtractTicketInfo.model_validate_json(response.message.content)
    logger.info(
        f"Extraction complete - Departure: {result.departure}, Destination: {
            result.destination}, Departure Date: {result.departure_date}, Return Date: {result.return_date}"
    )
    return result


def get_confirmation_message(source: str, destination: str, departure_date: datetime, return_date: Optional[datetime] = None) -> GenerateConfirmation:
    """Third LLM call to generate a confirmation message for the ticket booking"""
    logger.info("Generating confirmation message")

    # Some stub code to generate the ticket id
    ticket_id = str(random.randint(10000, 99999))

    # Prepare the context for the message
    flight_details = {
        "source": source,
        "destination": destination,
        "departure_date": departure_date.strftime("%B %d, %Y"),
        "return_date": return_date.strftime("%B %d, %Y") if return_date else None,
        "ticket_id": ticket_id
    }

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Generate a friendly confirmation message for a flight ticket booking. Include all relevant details. Return the output as JSON.",
            },
            {"role": "user", "content": str(flight_details)}
        ],
        options={"temperature": 0.0},
        format=GenerateConfirmation.model_json_schema()
    )

    result = GenerateConfirmation.model_validate_json(response.message.content)
    logger.info(
        f"Confirmation message generated successfully - {result.confirmation_message}")
    return result


user_input = input("Please enter your request: ")
logger.info(f"User input: {user_input}")
# user_input = "I want to book a ticket from New York to London on Mar 1 and return on Mar 5"

# -----------------------------------------------------#
# Execute the prompt chain                            #
# -----------------------------------------------------#
# --> First LLM call to determine if input is a new flight or airline ticket booking request
result = check_ticket_booking(user_input)
if result.is_ticket_booking:  # --> Gate: If the first LLM call indicates a ticket booking request the proceed to the next step
    # --> Second LLM call to extract ticket information like source, destination, and dates
    ticket_info = extract_ticket_info(user_input)
    if ticket_info.departure_date is None:
        depature_date = input("Please enter the departure date: ")
        ticket_info = extract_ticket_info(
            f'{user_input} and the departure date is {depature_date}')
        # Stub: Assume a function is called to create a ticket and generate a ticket ID
        ticket_id = "12345"
        get_confirmation_message(ticket_info.departure, ticket_info.destination,
                                 ticket_info.departure_date, ticket_info.return_date)
    else:
        get_confirmation_message(ticket_info.departure, ticket_info.destination,
                                 ticket_info.departure_date, ticket_info.return_date)
else:  # --> Gate: If the first LLM call indicates that the input is not a ticket booking request then exit
    logger.error("Sorry, I cannot help you with this request")
