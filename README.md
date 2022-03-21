# ratingsapp
Console line app for self-updating rating system

### Installation

```
python3 -m venv .venv
pip install -r requirements.txt
source .venv/bin/activate

pip install -e .
```

1. Crates a new virtual env
2. Install all required packages in this new virtual env
3. Activates this virtual env
4. Installs the 'frank' package

### Usage

Navigate to project folder and activate the virtual env. Use the app!
```
source .venv/bin/activate

frank list players
```

It is possible to install package in global python which should enable to use command frank from anywhere. Currently it throws me some error. And maybe it does not know how to find DATA folder if I am anywhere?