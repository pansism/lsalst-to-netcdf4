import platform
from lsalst2netcdf import LSALSTstack2NetCDF

if platform.system() == "Linux" or platform.system() == "Darwin":  # Darwin is Mac
    project_folder = "/Users/panosis/Dropbox/MyCodeRepository/sandbox/Data"
    datadir = "/Users/panosis/Dropbox/MyCodeRepository/sandbox/Data/lsalst_h5_sample"

elif platform.system() == "Windows":
    project_folder = r"C:\Users\User\Dropbox\MyCodeRepository\sandbox"
    datadir = r"C:\Users\User\Dropbox\MyCodeRepository\sandbox\Data\lsalst_h5_sample"

else:
    print("Unable to determine which OS you are using")

LSALSTstack2NetCDF(
    savedir=project_folder,
    savename="downscalingtest_8",
    h5dir=datadir,
    latN=38.525,
    latS=37.475,
    lonW=23.175,
    lonE=24.225,
)

