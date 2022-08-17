#pragma once

#include <maya/MTypeId.h> 
#include <maya/MPlug.h> 

#include <maya/MMatrix.h>
#include <maya/MDoubleArray.h>

#include <maya/MPxNode.h> 
#include <maya/MItGeometry.h>

#define DEFORMER_NAME "shell"

class shell : public MPxNode
{
public:
	shell();
	~shell() override;

    static void*   creator();
    static MStatus initialize();

    static MObject aInputGeom; // mesh
    static MObject aOutputGeom; // mesh
        
    static MObject aPosThickness; // float
    static MObject aNegThickness; // float
    static MObject aThickLoops; // int

    static const MTypeId id;
private:
    virtual MStatus compute(
        const MPlug& plug,
		MDataBlock& block
	);
};
