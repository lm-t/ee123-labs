#Parts of the code grabbed from David Barrat -- jp2-python
import numpy as np

def max_ndarray(mat):
    """
    Returns maximum value within a given 2D Matrix, otherwise 0
    Parameters
    ----------
    mat: numpy.ndarray
        matrix from which we want to compute the max value
    Returns
    -------
    int32:
        matrix maximum value
    """
    return np.amax(mat) if type(mat).__name__ == 'ndarray' else 0

def extract_rgb_coeff(img):
    'Extracts RGB coefficients of a given Image'
    (width, height) = img.size
    img = img.copy()

    mat_r = numpy.empty((width, height))
    mat_g = numpy.empty((width, height))
    mat_b = numpy.empty((width, height))

    for i in range(width):
        for j in range(height):
            (r, g, b) = img.getpixel((i, j))
            mat_r[i, j] = r
            mat_g[i, j] = g
            mat_b[i, j] = b

    # coeffs_r: cA,(cH,cV,cD)
    coeffs_r = pywt.dwt2(mat_r, 'haar')
    # coeffs_g: cA,(cH,cV,cD)
    coeffs_g = pywt.dwt2(mat_g, 'haar')
    # coeffs_b: cA,(cH,cV,cD)
    coeffs_b = pywt.dwt2(mat_b, 'haar')
    return (coeffs_r, coeffs_g, coeffs_b)

def img_from_dwt_coeff(coeff_dwt):
    '''
    This function will recreate an Image object, based on the generated from dwt coefficients
    '''
    #Channel Red
    (coeffs_r, coeffs_g, coeffs_b) = coeff_dwt
    cARed = numpy.array(coeffs_r[0])
    (width, height) = coeffs_r.shape
    cHRed = numpy.array(coeffs_r[1][0])
    cVRed = numpy.array(coeffs_r[1][1])
    cDRed = numpy.array(coeffs_r[1][2])
    #Channel Green
    cAGreen = numpy.array(coeffs_g[0])
    cHGreen = numpy.array(coeffs_g[1][0])
    cVGreen = numpy.array(coeffs_g[1][1])
    cDGreen = numpy.array(coeffs_g[1][2])
    #Channel Blue
    cABlue = numpy.array(coeffs_b[0])
    cHBlue = numpy.array(coeffs_b[1][0])
    cVBlue = numpy.array(coeffs_b[1][1])
    cDBlue = numpy.array(coeffs_b[1][2])

    # maxValue per channel par matrix
    cAMaxRed = max_ndarray(cARed)
    cAMaxGreen = max_ndarray(cAGreen)
    cAMaxBlue = max_ndarray(cABlue)

    cHMaxRed = max_ndarray(cHRed)
    cHMaxGreen = max_ndarray(cHGreen)
    cHMaxBlue = max_ndarray(cHBlue)

    cVMaxRed = max_ndarray(cVRed)
    cVMaxGreen = max_ndarray(cVGreen)
    cVMaxBlue = max_ndarray(cVBlue)

    cDMaxRed = max_ndarray(cDRed)
    cDMaxGreen = max_ndarray(cDGreen)
    cDMaxBlue = max_ndarray(cDBlue)

    # Image object init
    dwt_img = Image.new('RGB', (width*2, height*2), (0, 0, 20))
    #cA reconstruction
    for i in range(width):
        for j in range(height):
            R = cARed[i][j]
            R = (R/cAMaxRed)*160.0
            G = cAGreen[i][j]
            G = (G/cAMaxGreen)*85.0
            B = cABlue[i][j]
            B = (B/cAMaxBlue)*100.0
            new_value = (int(R), int(G), int(B))
            dwt_img.putpixel((i, j), new_value)
    #cH reconstruction
    for i in range(width):
        for j in range(height):
            R = cHRed[i][j]
            R = (R/cHMaxRed)*160.0
            G = cHGreen[i][j]
            G = (G/cHMaxGreen)*85.0
            B = cHBlue[i][j]
            B = (B/cHMaxBlue)*100.0
            new_value = (int(R), int(G), int(B))
            dwt_img.putpixel((i+width, j), new_value)
    #cV reconstruction
    for i in range(width):
        for j in range(height):
            R = cVRed[i][j]
            R = (R/cVMaxRed)*160.0
            G = cVGreen[i][j]
            G = (G/cVMaxGreen)*85.0
            B = cVBlue[i][j]
            B = (B/cVMaxBlue)*100.0
            new_value = (int(R), int(G), int(B))
            dwt_img.putpixel((i, j+height), new_value)
    #cD reconstruction
    for i in range(width):
        for j in range(height):
            R = cDRed[i][j]
            R = (R/cDMaxRed)*160.0
            G = cDGreen[i][j]
            G = (G/cDMaxGreen)*85.0
            B = cDBlue[i][j]
            B = (B/cDMaxBlue)*100.0
            new_value = (int(R), int(G), int(B))
            dwt_img.putpixel((i+width, j+height), new_value)
return dwt_img
