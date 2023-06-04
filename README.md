# API Reference
https://github.com/Kaggle/kaggle-api

# Usage
```
# Download
poetry run kaggle competitions download -p {dir} {competition}

# Pull kernel with metadata
# Create notebook in the kaggle website first.
poetry run kaggle kernels pull keewoongnam/{competition} -m

# Submit
poetry run kaggle competitions submit -f answer.csv -m '{message}' {competition}

# Check
poetry run kaggle competitions submissions {competition}

# Push kernal
poetry run kaggle kernels push -p {competition}
```

