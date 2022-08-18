#include <maya/MTypeId.h> 

#include <maya/MTime.h> 
#include <maya/MMatrix.h>
#include <maya/MDoubleArray.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MFloatPointArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MTransformationMatrix.h>

#include <maya/MItGeometry.h>
#include <maya/MItMeshVertex.h>
#include <maya/MItSurfaceCV.h>

#include <maya/MFnNurbsSurface.h>
#include <maya/MFnMesh.h>

#include <maya/MFnUnitAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>

#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>  // std::sort
#include <numeric>    // std::iota

#include "shell.h"
#include "maxShellAlg.h"

#include "xxhash.h"


struct {
    MIntArray faces;
    MIntArray counts;
} MeshType;

typedef std::pair<int, int> Edge;


MObject shell::aPosThickness;
MObject shell::aNegThickness;
MObject shell::aThickLoops;

MObject shell::aInputGeom;
MObject shell::aOutputGeom;







const MTypeId shell::id(0x00122713);
void* shell::creator() {return new shell();}
MStatus shell::initialize() {
    MStatus stat;
    MFnNumericAttribute fnNum;
    MFnTypedAttribute fnTyped;
    MFnUnitAttribute fnUnit;



    aPosThickness = fnUnit.create("posThickness", "pt", MFnUnitAttribute::kDistance, 0.1);
    fnUnit.setStorable(true);
    fnUnit.setKeyable(true);
    stat = addAttribute(aPosThickness);

    aNegThickness = fnUnit.create("negThickness", "nt", MFnUnitAttribute::kDistance, 0.1);
    fnUnit.setStorable(true);
    fnUnit.setKeyable(true);
    stat = addAttribute(aNegThickness);

    aThickLoops = fnNum.create("thickLoops", "tl", MFnNumericData::kInt, 1);
    fnNum.setMin(1);
    fnNum.setStorable(true);
    fnNum.setKeyable(true);
    stat = addAttribute(aThickLoops);



    aInputGeom = fnTyped.create("inputGeom", "ig", MFnData::kMesh);
    stat = addAttribute(aInputGeom);

    aOutputGeom = fnTyped.create("outputGeom", "og", MFnData::kMesh);
    stat = addAttribute(aOutputGeom);


    std::vector<MObject *> iobjs = {
        &aPosThickness, &aNegThickness, &aThickLoops, &aInputGeom
    };

    std::vector<MObject *> oobjs = {&aOutputGeom};

    for (auto &ii : iobjs) {
        for (auto &oo : oobjs) {
            attributeAffects(*ii, *oo);
        }
    }

    return MStatus::kSuccess;
}


shell::shell() {}
shell::~shell() {}


MStatus shell::compute(
	const MPlug& plug,
	MDataBlock& dataBlock
){
	MStatus status;

    if (plug != aOutputGeom){
        // Only the output geom plug matters
        return MStatus::kUnknownParameter;
    }

	MDataHandle hPos = dataBlock.inputValue(aPosThickness);
	float pos = (float)hPos.asDouble();

	MDataHandle hNeg = dataBlock.inputValue(aNegThickness);
	float neg = (float)hNeg.asDouble();

	MDataHandle hLoops = dataBlock.inputValue(aThickLoops);
	int loops = hLoops.asInt();

	MDataHandle hInput = dataBlock.inputValue(aInputGeom);
	MDataHandle hOutput = dataBlock.outputValue(aOutputGeom);

    MObject inMesh = hInput.asMesh();
    MFnMesh inFnMesh(inMesh);

    MIntArray count, faces;
    inFnMesh.getVertices(count, faces);

    // Hash the input topology so I can update if it changes
    XXH64_hash_t cHashChk = XXH3_64bits(&(count[0]), count.length() * sizeof(int));
    XXH64_hash_t fHashChk = XXH3_64bits(&(faces[0]), faces.length() * sizeof(int));
    bool needNewTopo = (loops != loopStore) || (cHashChk != cHash) || (fHashChk != fHash);
    if (needNewTopo) {
        bVerts = shellTopo(faces, count, inFnMesh.numVertices(), loops, newFaces, newCount);
        cHash = cHashChk;
        fHash = fHashChk;
        loopStore = loops;
    }

    // Do the point positions
    MPointArray verts;
    MFloatVectorArray normals;
    inFnMesh.getPoints(verts);
    inFnMesh.getVertexNormals(false, normals);
    auto newVerts = shellGeo(verts, normals, bVerts, loops, neg, pos);

    if (needNewTopo) {
        // Create a new mesh MObject
        hOutput.set(inMesh);
        outMesh = hOutput.asMesh();
        MFnMesh meshFn(outMesh);
        meshFn.createInPlace(newVerts.length(), newCount.length(), newVerts, newCount, newFaces);
    }
    else {
        hOutput.set(outMesh);
        MFnMesh meshFn(outMesh);
        meshFn.setPoints(newVerts);
    }

    hOutput.setClean();
    return status;
}




