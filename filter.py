import os
import shutil

inputPath = 'flowers'
outputPath = 'data'

if not os.path.exists(outputPath):
        os.mkdir(outputPath)

for fileName in os.listdir(inputPath):
    className = fileName[:fileName.find('.')]
    if not os.path.exists(os.path.join(outputPath, className)):
        os.mkdir(os.path.join(outputPath, className))
    shutil.move(os.path.join(inputPath, fileName), os.path.join(outputPath, className, fileName))