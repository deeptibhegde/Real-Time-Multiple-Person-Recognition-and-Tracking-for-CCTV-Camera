import sys
import subprocess

def fn_get_txt_sysarg():
    """Harvest a single (the only expected) command line argument"""
    try:
        return sys.argv[1]    # str() would be redundant here
    except:
        ErrorMsg = 'Message from fn_get_txt_sysarg() in Script (' + sys.argv[0] + '):\n' + '\tThe Script did not receive a command line argument'
        sys.exit(ErrorMsg)

def Open_Win_Explorer_and_Select_Fil(filepath):
    # harvested from: https://stackoverflow.com/questions/281888/open-explorer-on-a-file
    Popen_arg = 'explorer /select,"' + filepath + "'"    # str() is redundant here also
    subprocess.Popen(Popen_arg)

if __name__ == '__main__': 
    filepath = fn_get_txt_sysarg()
    Open_Win_Explorer_and_Select_Fil(filepath)