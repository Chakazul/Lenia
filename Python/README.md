### How-To

##### Linux (arch):
(in Lenia/Python folder)
1. Make virtual environment:
```bash
virtualenv *env_name*
```
2. Activate your virtual environment:
```bash
source *env_name*/bin/activate
```
3. Install libraries:
```bash
pip install -r requirements.txt
```

------------


### Trouble-shoot:

**[Error]** ImportError: libtk8.6.so: cannot open shared object file: No such file or directory

**[Solution]** Install tk


```bash
sudo pacman -S tk
```
or
```bash
sudo apt-get install tk
```

or if Tkinter module not found on Ubuntu:

```bash
sudo apt-get install python3-tk
```

For python 3.6:
```bash
sudo apt-get install python3.6-tk
```
**[Error]** libf77blas.so.3: cannot open shared object file: No such file or dircetory
**[Solution]** Install libatlas3-base
```bash
sudo apt install libatlas3-base
```
