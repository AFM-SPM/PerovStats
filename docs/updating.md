# Updating PerovStats

### As new updates come in you will need to pull PerovStats from GitHub again to get these changes. This is a fairly simple process:

- First, check that you have any files that you want to keep (e.g. output files) saved outsite of the `/PerovStats/` folder.

- Clear all outputs in both notebooks (and save), this avoids merge conflicts when pulling the new changes.

- Ensure your virutal environment is running. You can confirm this by checking for a green `(venv)` on the left of your terminal.

    ![(venv) showing in terminal](./images/venv_check.jpg)

    - If not, navigate to the folder your virtual environment is in and type:

        - **Windows:** `venv\Scripts\activate`
        - **macOS/Linux:** `source venv/bin/activate`

- Navigate to the `/PerovStats/` folder in the same terminal

- Type `git pull origin main` in this terminal - this should download all new content/ updates
