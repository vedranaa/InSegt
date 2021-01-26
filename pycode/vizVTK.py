import vtk
import tifffile


def visgray(vol,
           scalarOpacity = {0: 0.0, 255: 1.0}, # default to transparent black and opaque white
           gradientOpacity = {0: 0.0, 100: 1.0}, # transparent low-grad regions
           colorTransfer = {0: (0.0, 0.0, 0.0), 255: (1.0, 1.0, 1.0)}, # zero is black
           backgroundColor = (1.0, 1.0, 1.0), # white background
           windowSize = None, 
           windowName = None
           ):
    '''
    
    Parameters
    ----------
    vol : 3D numpy aray of type uint8 (check what happens if not)

    '''
    
    # Store data as VTK-image.
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(vol.tobytes(), vol.size)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetWholeExtent(0, vol.shape[2]-1, 0, vol.shape[1]-1, 0, vol.shape[0]-1)
    dataImporter.SetDataExtentToWholeExtent()
    
    # Ray-cast mapper.
    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    
    # Preparing to collect volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    
    # Opacity  transfer function.
    if scalarOpacity is not None:
        scalarOpacityFunc = vtk.vtkPiecewiseFunction()
        for point in scalarOpacity:
            scalarOpacityFunc.AddPoint(point, scalarOpacity[point])
        volumeProperty.SetScalarOpacity(scalarOpacityFunc)
        
    # Gradient opacity function (decrease opacity in constant-intensity regions).
    if gradientOpacity is not None:
        gradientOpacityFunc = vtk.vtkPiecewiseFunction()
        for point in gradientOpacity:
            gradientOpacityFunc.AddPoint(point, gradientOpacity[point])
        volumeProperty.SetGradientOpacity(gradientOpacityFunc)
    
    
    # Color transfer function.
    if colorTransfer is not None:
        colorTransferFunc = vtk.vtkColorTransferFunction()
        for point in colorTransfer:
            colorTransferFunc.AddRGBPoint(point, colorTransfer[point][0], 
                    colorTransfer[point][1], colorTransfer[point][2])
        volumeProperty.SetColor(colorTransferFunc)
       
    # Volume pairs properties and mapper.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    
    # Renderer.
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    if backgroundColor is not None:
        renderer.SetBackground(backgroundColor[0], backgroundColor[1], 
                               backgroundColor[2])

    # Window.
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    if windowSize is not None:
        renderWin.SetSize(windowSize[0], windowSize[1])
    if windowName is not None:
        renderWin.SetWindowName(windowName)

    # Interactor.
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renderInteractor.SetRenderWindow(renderWin)
    
    # Showtime.
    renderInteractor.Initialize()    
    renderWin.Render()
    renderInteractor.Start()

if __name__ == '__main__':    
    filename = '../data/nerves_part.tiff'
    
    scalarOpacity = {0: 0.9, 100: 0.5, 255: 0.0} # transparent white
    gradientOpacity = {0: 0.0, 90: 0.5, 100: 1} # transparent low-grad regions
    colorTransfer = {
        0: (0.0, 0.0, 0.0),
        255: (1.0, 1.0, 1.0)} # zero is black
    
    
    V = tifffile.imread(filename)
    
    
    visgray(V[:100,:100,:100],
            scalarOpacity=scalarOpacity, 
            gradientOpacity=gradientOpacity, 
            colorTransfer = colorTransfer,
            windowSize = (750,750),
            windowName = 'Cool wiz')

