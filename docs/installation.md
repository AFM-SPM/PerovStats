# PerovStats Installation instructions

## Python 3.11

### PerovStats requires python versoin 3.11 to run. If you are not sure which version you have installed:

- Open the command prompt or terminal:

    - **Windows:** Press the windows key, type `cmd` and press **Enter**
    - **macOS:** Press **Command (⌘) + Spacebar**, type `Terminal` and press **Enter**
    - **Linux:** Press **Ctrl + Alt + T**

- Check your python version (if installed)

    Type `py --list` in the window that opens to get a list of all installed python versions. Python 3.11 will look like this: `-V:3.11          Python 3.11 (64-bit)`.

    If you do not see python 3.11 or you do not have any version on your system you will have to download it from [here](https://www.python.org/downloads/release/python-3110/) (select the installer from the bottom of the page that matches your operating system).

    Python 3.11 will now be on your system and ready to use. Confirm this by typing the `py --list` command again.

---

## Repo cloning and virtual environment

- Firstly open a command prompt or terminal in the folder you would like the PerovStats program in. You can navigate through folders in the terminal by typing `cd [childFolder]` or return to the last parent folder with `cd ..`

- Once inside the correct folder, clone the github repo by entering this command:

    `git clone https://github.com/AFM-SPM/PerovStats.git`

- When it has finishing downloading, type `cd PerovStats` to enter the project directory

- A virtual environment is best used for storing and running PerovStats. Inside the main project folder you just downloaded:

    - Create the virtual environment with:

        - **Windows:** `py -3.11 -m venv venv`
        - **macOS/Linux:** `python3.11 -m venv venv`

        This will create a new folder within the project called `venv`.

    - Now you can start up the environment with:

        - **Windows:** `venv\Scripts\activate`
        - **macOS/Linux:** `source venv/bin/activate`

        In both cases you will see `(venv)` appear on the left of your command line.

- The requirments and dependencies must now be installed into the virtual environment. In the same directory still, type `pip install -r requirements.txt`

    *Note: If a **yellow** warning appears about 'upgrading pip' this can be ignored and will not stop PerovStats from working.*

- **If you would like to run PerovStats from the command line rather than/ as well as from jupyter notebooks:**

    - Run the command `pip install -e .`

#### PerovStats should now be ready to run. Refer to `usage.md` for instructions on the next steps
