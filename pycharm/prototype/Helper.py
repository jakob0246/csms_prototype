from colorama import Fore
from colorama import Style


def print_warning(text):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")