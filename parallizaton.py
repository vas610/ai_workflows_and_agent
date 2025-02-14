#!.venv/bin/python3
# Parallelization Example
import logging
import os
from datetime import datetime
from typing import Optional, Literal
import asyncio
import random
import json

from ollama import ChatResponse, chat, AsyncClient
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
MODEL_NAME = "phi4"

# Load existing data from the JSON file into a dictionary if the file exists and has some data
if (
    os.path.exists("flight_booking_details.json")
    and os.path.getsize("flight_booking_details.json") > 0
):
    with open("flight_booking_details.json", "r") as f:
        all_booking_details = json.load(f)
else:
    all_booking_details = {}

logger.debug(f"Existing booking details: {all_booking_details}")

# -----------------------------------------------------#
# Define the data models for each stage using Pydantic #
# -----------------------------------------------------#


class CheckTicketBooking(BaseModel):
    """Check if this request is about booking a new ticket"""

    description: str = Field(description="Raw description of the user input")
    is_ticket_booking: bool = Field(
        description="Whether this text describes booking a flight or airline ticket"
    )
    new_or_modify: Literal["new", "modify"] = Field(
        description="Whether this is a new booking or a modification of an existing booking"
    )


class ExtractTicketInfo(BaseModel):
    """Extract ticket information like source, destination, and dates"""

    source: str = Field(description="Departure location of the flight")
    destination: str = Field(description="Destination location of the flight")
    departure_date: Optional[datetime] = Field(
        default_factory=lambda: None, description="Departure date of the flight"
    )
    return_date: Optional[datetime] = Field(
        default_factory=lambda: None,
        description="Date when trip ends (if applicable else None)",
    )
    ticket_id: Optional[int] = Field(
        default_factory=lambda: None, description="Ticket ID for the booking"
    )


class FlightBookingConfirmation(BaseModel):
    """Generate a confirmation message for the ticket booking"""

    confirmation_message: str = Field(
        "A confirmation message for the ticket booking that includes the source, destination, dates and ticket ID"
    )


class HotelBookingConfirmation(BaseModel):
    """Generate a confirmationm message for the hotel booking"""

    hotel_confirmation_message: str = Field(
        "Confirmation message for the hotel booking"
    )


class CombinedBookingConfirmation(BaseModel):
    """Generate a confirmationm message that combines both flight and hotel booking information"""

    combined_confirmation_message: str = Field(
        "Confirmation message for the entire trip including both flight and hotel booking."
    )


# -----------------------------------------------------#
# Define the functions for making the calls to LLMs    #
# -----------------------------------------------------#


async def generate_ticket_id() -> int:
    """Generate a random ticket ID for the booking"""
    return random.randint(10000, 99999)


async def check_ticket_booking(user_input: str) -> CheckTicketBooking:
    """First LLM call to determine if input is a new flight or airline ticket booking request"""
    logger.info(
        "LLM-call-1: Start checking if the input describes booking a flight new ticket"
    )
    logger.debug(f"Input text: {user_input}")

    response = await AsyncClient().chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"Determine if the given text requests for booking a flight ticket. \
                                Also, determine if it is a new booking or a modification of an existing booking. Return the output as JSON.",
            },
            {"role": "user", "content": user_input},
        ],
        options={"temperature": 0},
        # -- Structured output using the given data model
        format=CheckTicketBooking.model_json_schema(),
    )
    result = CheckTicketBooking.model_validate_json(response.message.content)
    logger.info(
        f"Flight ticket booking check complete - Is ticket booking: {
            result.is_ticket_booking}"
    )
    return result


async def extract_ticket_info(user_input: str) -> ExtractTicketInfo:
    """Second LLM call to extract ticket information like source, destination, and dates"""
    logger.info("LLM-call-2: Start extracting ticket information")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    today_date_for_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"{today_date_for_context}.\
                                Extract ticket information like source, destination, dates and return  as JSON .\
                If no source is mentioned, then set it as None.\
                If no destination is mentioned, then set it as None.\
                If no return date is not mentioned, then set it as null.\
                Do not set departure date and return date unless specified in the input.",
            },
            {"role": "user", "content": user_input},
        ],
        options={"temperature": 0},
        # --> Structured output using the given data model
        format=ExtractTicketInfo.model_json_schema(),
    )
    result = ExtractTicketInfo.model_validate_json(response.message.content)
    logger.info(
        f"Extraction complete - Departure: {result.source}, Destination: {
            result.destination}, Departure Date: {result.departure_date}, Return Date: {result.return_date}"
    )
    return result


async def get_confirmation_message(
    new_or_modify: str,
    source: str,
    destination: str,
    departure_date: datetime,
    return_date: Optional[datetime] = None,
    ticket_id: Optional[int] = None,
) -> FlightBookingConfirmation:
    """Third LLM call to generate a confirmation message for the ticket booking"""
    logger.info("LLM-call-3: Generating confirmation message")

    if ticket_id is None:
        ticket_id = await generate_ticket_id()

    # Prepare the context for the message
    flight_details = {
        "source": source,
        "destination": destination,
        "departure_date": departure_date.strftime("%B %d, %Y"),
        "return_date": return_date.strftime("%B %d, %Y") if return_date else None,
        "ticket_id": ticket_id,
    }

    logger.info(f"Flight details: {flight_details}")

    # Store the flight details in a file as JSON
    ticket_id_str = str(flight_details["ticket_id"])
    if all_booking_details.get(ticket_id_str) is not None:
        all_booking_details[ticket_id_str].update(flight_details)
    else:
        all_booking_details[flight_details["ticket_id"]] = flight_details
    with open("flight_booking_details.json", "w") as f:
        json.dump(all_booking_details, f)

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"Generate a friendly confirmation message for a flight ticket booking.\
                      Include all relevant details. \
                      Write the confirmation message based on new or modified action_type.\
                      action_type = {new_or_modify}.\
                      Include the ticket ID in the confirmation messsage.\
                      Return the output as JSON.",
            },
            {"role": "user", "content": str(flight_details)},
        ],
        options={"temperature": 0},
        format=FlightBookingConfirmation.model_json_schema(),
    )

    result = FlightBookingConfirmation.model_validate_json(response.message.content)
    logger.debug(
        f"Flight Booking - {
            result.confirmation_message}"
    )
    return result


async def modify_existing_booking(user_input: str, ticket_id: int) -> ExtractTicketInfo:
    """Routed: Second LLM call to extract ticket information like source, destination, and dates from an existing booking"""
    logger.info("LLM-call-2: Start extracting ticket modification information")
    logger.info(f"Input text: {user_input}")

    today = datetime.now()
    today_date_for_context = f"Today is {today.strftime('%A, %B %d, %Y')}."
    existing_booking_context = f"Existing booking info: {
        all_booking_details.get(str(ticket_id), None)}"
    logger.debug(existing_booking_context)

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"{today_date_for_context}.\
                                {existing_booking_context}.\
                                 Modify the source, destination and dates in the existing booking info based on the new input.\
                                 Retain the existing values if no change has been requested for those attributes.\
                                 Return as JSON.",
            },
            {"role": "user", "content": user_input},
        ],
        options={"temperature": 0},
        # --> Structured output using the given data model
        format=ExtractTicketInfo.model_json_schema(),
    )
    result = ExtractTicketInfo.model_validate_json(response.message.content)

    logger.info(
        f"Modification details extraction complete - Departure: {result.source}, Destination: {
            result.destination}, Departure Date: {result.departure_date}, Return Date: {result.return_date}, Ticket ID: {ticket_id}"
    )
    return result


async def flight_ticket_booking(ticket_id: int, user_input: str):
    check_ticket = await check_ticket_booking(user_input)
    if (
        check_ticket.is_ticket_booking
    ):  # --> Gate: If the first LLM call indicates a ticket booking request the proceed to the next step
        # --> Second LLM call to extract ticket information like source, destination, and dates
        if check_ticket.new_or_modify == "new":
            logger.info("This is a new ticket booking request")
            ticket_info = await extract_ticket_info(user_input)
            if ticket_info.departure_date is None:
                depature_date = input("Please enter the departure date: ")
                ticket_info = await extract_ticket_info(
                    f"{user_input} and the departure date is {depature_date}"
                )
                # --> Third LLM call to generate a confirmation message for the ticket booking
                flight_confirmation = await get_confirmation_message(
                    check_ticket.new_or_modify,
                    ticket_info.source,
                    ticket_info.destination,
                    ticket_info.departure_date,
                    ticket_info.return_date,
                    ticket_id,
                )
                return flight_confirmation
            else:
                # --> Third LLM call to generate a confirmation message for the ticket booking
                flight_confirmation = await get_confirmation_message(
                    check_ticket.new_or_modify,
                    ticket_info.source,
                    ticket_info.destination,
                    ticket_info.departure_date,
                    ticket_info.return_date,
                    ticket_id,
                )
                return flight_confirmation
        elif check_ticket.new_or_modify == "modify":
            logger.info("This is a modification request for an existing ticket")
            # ticket_id = input("Please enter the ticket ID followed by the modification needed: ")
            modified_ticket_info = modify_existing_booking(user_input, ticket_id)
            flight_confirmation = await get_confirmation_message(
                check_ticket.new_or_modify,
                modified_ticket_info.source,
                modified_ticket_info.destination,
                modified_ticket_info.departure_date,
                modified_ticket_info.return_date,
                ticket_id,
            )
            return flight_confirmation
        else:
            logger.error("Invalid new_or_modify value")
    else:  # --> Gate: If the first LLM call indicates that the input is not a ticket booking request then exit
        logger.error("Sorry, I cannot help you with this request")


async def hotel_booking(ticket_id: int, user_input: str) -> HotelBookingConfirmation:
    """Book a hotel room based on the destination and dates"""
    logger.info("LLM-call-1: Start extracting hotel booking information")

    today = datetime.now()
    today_date_for_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    response = await AsyncClient().chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"{today_date_for_context}.\
                    Ticket Id is {ticket_id}.\
                    Based on the user input, extract the desitnation, departure date and return date.\
                    Use the departure date as the check-in date and the return date as the check-out date.\
                    Check out date should be at least 1 day after the check in date.\
                    Check out date should be the same as the return date if mentioned.\
                    Generate a confirmation message for the hotel booking in a friendly manner.\
                    The confirmation message should contain the destination, check-in date, check-out date and ticket id\
                    Return the output as JSON.",
            },
            {"role": "user", "content": user_input},
        ],
        options={"temperature": 0},
        format=HotelBookingConfirmation.model_json_schema(),
    )
    result = HotelBookingConfirmation.model_validate_json(response.message.content)
    logger.debug(f"Hotel Booking: {result.hotel_confirmation_message}")
    return result


async def combined_booking_confirmation(
    flight_confirmation: int, hotel_confirmation: str
) -> CombinedBookingConfirmation:
    """Generate a confirmationm message that combines both flight and hotel booking information"""
    logger.info("LLM-call-3: Generate confirmation message")

    today = datetime.now()
    today_date_for_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"{today_date_for_context}.\
                    Generate a confirmation message for the hotel booking in a friendly manner.\
                    The confirmation message should contain the information in the following manner:\
                    <a-friendly-greeting>.\
                    **Ticket ID**\
                    Ticket ID: <ticket_id>.\
                    **Flight ticket details**\
                    Source: <source>\
                    Destination: <destination>\
                    Departure Date: <departure_date>\
                    Return Date: <return_date>\
                    **Hotel booking details**\
                    Destination: <destination>\
                    Check-in Date: <check_in_date>\
                    Check-out Date: <check_out_date>\
                    <-a-friendly-closing-remark>.\
                    ",
            },
            {
                "role": "user",
                "content": f"Flight Booking Confirmation: {flight_confirmation.confirmation_message}\nHotel Booking Confirmation: {hotel_confirmation.hotel_confirmation_message}",
            },
        ],
        options={"temperature": 0},
        format=CombinedBookingConfirmation.model_json_schema(),
    )
    result = CombinedBookingConfirmation.model_validate_json(response.message.content)
    logger.info(f"Booking Confirmation - {result.combined_confirmation_message}")
    return result


# -----------------------------------------------------#
# Execute the parallel workflow                        #
# -----------------------------------------------------#

# Get user input
user_input = "I want to book a flight ticket from New York to London on Mar 10 and return on Mar 25"

async def main():
    # Generate a ticket ID
    ticket_id = await generate_ticket_id()
    # Book a flight and hotel room in parallel
    flight_confirmation, hotel_confirmation = await asyncio.gather(
        flight_ticket_booking(ticket_id, user_input),
        hotel_booking(ticket_id, user_input),
    )
    # Combine the booking confirmations
    await combined_booking_confirmation(flight_confirmation, hotel_confirmation)

# Run Workflow
asyncio.run(main())
