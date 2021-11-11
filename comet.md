# Use
Pass `--comet <project name> <experiment name>` to `train.py`.
- Project need not exist yet, a new one will be automatically created. HOWEVER
  for similar work, place all experiments under the same project.
- Experiment name need not be unique, comet gives each a unique ID. Of course it's a good
  idea to give it a useful name regardless.

# API Key
1. Go to https://www.comet.ml/wblount/settings/account.
2. Under the "Developer Information" section, obtain your API key.
3. Make a file called `.comet.config` in either (see [here](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration) for format):
   1. Home directory (i.e `~/`) OR
   2. Project directory (i.e, same as this file)
4. Now when you run `train.py` as above, comet will read your API key and tie
   the experiment to you automatically.
