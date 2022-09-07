#pragma once

#include <maya/MTypeId.h> 
#include <maya/MPlug.h> 

#include <maya/MMatrix.h>
#include <maya/MDoubleArray.h>
#include <maya/MFloatArray.h>

#include <maya/MPxNode.h> 
#include <maya/MItGeometry.h>
#include "xxhash.h"

#include <vector>

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
    static MObject aUvThickness; // float
    static MObject aThickLoops; // int

    static const MTypeId id;
private:
    virtual MStatus compute(
        const MPlug& plug,
		MDataBlock& block
	);

    XXH64_hash_t cHash = 0;
    XXH64_hash_t fHash = 0;
    std::vector<std::vector<size_t>> grid;
    MIntArray newCount, newFaces, newUvCount, newUvFaces;
    MFloatArray newUArray, newVArray;
    MObject outMesh;
    int loopStore = 0;
};
