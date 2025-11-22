# How to clone the template
This is a template repo. If you want to create a new repo out of it then:

## GUI
1. Visit https://git.fit.vutbr.cz/ or https://github.com/BUTSpeechFIT/ and start to create your new repo
2. There is `Template` select box at _git.fit.vutbr.cz_ where you can select this repo and then
  - Check Git content to copy the content of the template. We are interested just in the YAMLs 
3. There is no import option at _github.com_

## CMD
1. Clone the repo `git clone git@git.fit.vutbr.cz:szoke/repo_template_python.git` / `git clone git@github.com:BUTSpeechFIT/repo_template_python.git`
2. Create your repo `git init sample`
3. Copy the yamls `cp ./repo_template_python/*.yaml ./sample/`

# How to activate the hooks
1. Install the pre-commit tool using `pip install pre-commit` in your conda env (if you already do not have one).
2. Select your configuration according to your python version `cp .pre-commit-config-py310.yaml .pre-commit-config.yaml`
3. Run `pre-commit install --hook-type pre-commit --hook-type pre-push` to install the hooks
4. Thats all.

# TODOs
- Can we get rid of python version dependency? (to have one yaml serve to all?)
- Is there simpler way of how to do this exercise using command line?

# Notes & sources:
https://pre-commit.com/

There is a way how to automatically install pre-commit to your newly created repo from your local settings:
https://pre-commit.com/#automatically-enabling-pre-commit-on-repositories
but this was not tested.

If you need to change the Python version, edit the yaml and update the expected Python version.

