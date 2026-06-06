import fastf1


def load_race_session(
    year=2025,
    grand_prix="Abu Dhabi",
    session_type="R"
):

    session = fastf1.get_session(
        year,
        grand_prix,
        session_type
    )

    session.load()

    return session


def get_driver_laps(
    session,
    driver
):

    return session.laps.pick_drivers(
        driver
    )