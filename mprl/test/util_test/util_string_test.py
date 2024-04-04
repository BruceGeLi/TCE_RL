import mprl.util as util


def test_print():
    """Function for testing print"""

    # Test for print_line
    util.print_line_title("Test for print_line")
    util.print_line()
    util.print_line("=", 3)
    util.print_line("*")
    util.print_line(before=1, after=1)
    util.print_line(length=50)
    # util.print_line("=", 5.3)  # invalid length, will fail

    # Test for print_line_title
    util.print_line_title("Test for print_line_title")
    util.print_line_title("Some middle title", )
    util.print_line_title("Some left title", middle=False)

    # Test for print_wrap_title
    util.print_line_title("Test for print_wrap_title")
    util.print_wrap_title("Some wrapped title")
    util.print_wrap_title("Some wrapped title2", wrap=2)
    util.print_wrap_title("Some wrapped title#", char="#", wrap=2)
    # End of function test_print


def test_print_table():
    util.print_wrap_title("test_print_table")
    headers = ["#", "t-d", "ctx", "pred", "Use as"]
    data = [["1.", "T", "T", "T", "[enc, pred] * [time, value]"],
            ["2.", "T", "T", "F", "[enc] * [time, value]"],
            ["3.", "T", "F", "T", "[pred] * [time, value]"],
            ["4.", "F", "T", "F", "[enc] * [value]"],
            ["5.", "F", "F", "T", "[pred] * [value]"]]
    util.print_table(tabular_data=data,
                     headers=headers)


def test_get_formatted_date_time():
    util.print_wrap_title("test_get_formatted_date_time")
    print(util.get_formatted_date_time())


def main():
    test_print()
    test_print_table()
    test_get_formatted_date_time()


if __name__ == "__main__":
    main()
