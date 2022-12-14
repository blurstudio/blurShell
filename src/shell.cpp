#include <maya/MTypeId.h> 

#include <maya/MTime.h> 
#include <maya/MString.h> 
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
#include <maya/MFnMeshData.h>

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






MFloatPointArray shellVertGridPosMaya(
    const MFloatPointArray& verts,
    const MFloatVectorArray& normals,
    const std::vector<std::vector<size_t>>& grid,
    float innerOffset,
    float outerOffset
){
    uint numBridgeSegs = (uint)grid.size() - 1;
    auto& bVerts = grid[0];
    auto& eVerts = grid[numBridgeSegs];

    uint bVertCount = (numBridgeSegs - 1) * (uint)bVerts.size();

    MFloatPointArray ret;
    uint r = 0;
    ret.setLength(2 * verts.length() + bVertCount);

    for (uint i = 0; i < verts.length(); ++i){
        ret[r++] = verts[i] + normals[i] * outerOffset;
    }
    for (uint i = 0; i < verts.length(); ++i){
        ret[r++] = verts[i] - normals[i] * innerOffset;
    }

    for (uint segIdx = 1; segIdx < numBridgeSegs; ++segIdx){
        float perc = (float)segIdx / (float)(grid.size() - 1);
        for (uint vIdx = 0; vIdx < grid[segIdx].size(); ++vIdx){
            uint tar = (uint)grid[segIdx][vIdx];
            uint iSrc = (uint)bVerts[vIdx];
            uint oSrc = (uint)eVerts[vIdx];
            auto vi = ret[iSrc];
            auto vo = ret[oSrc];
            auto vip = vi * (1.0f - perc);
            auto vop = vo * perc;
            auto rr = vip + vop;
            ret[tar] = rr;
        }
    }
    return ret;
}









struct {
    MIntArray faces;
    MIntArray counts;
} MeshType;

typedef std::pair<size_t, size_t> Edge;


MObject shell::aPosThickness;
MObject shell::aNegThickness;
MObject shell::aUvThickness;
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

    aUvThickness = fnUnit.create("uvThickness", "uvt", MFnUnitAttribute::kDistance, 0.1);
    fnUnit.setStorable(true);
    fnUnit.setKeyable(true);
    stat = addAttribute(aUvThickness);

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
        &aPosThickness, &aNegThickness, &aUvThickness, &aThickLoops, &aInputGeom
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

	MDataHandle hUv = dataBlock.inputValue(aUvThickness);
	float uvOffset = (float)hUv.asDouble();

	MDataHandle hLoops = dataBlock.inputValue(aThickLoops);
	int loops = hLoops.asInt();

	MDataHandle hInput = dataBlock.inputValue(aInputGeom);
    MObject inMesh = hInput.asMesh();
    MFnMesh inFnMesh(inMesh);

    MIntArray count, faces;
    inFnMesh.getVertices(count, faces);

    // Hash the input topology so I can update if it changes
    XXH64_hash_t cHashChk = XXH3_64bits(&(count[0]), count.length() * sizeof(int));
    XXH64_hash_t fHashChk = XXH3_64bits(&(faces[0]), faces.length() * sizeof(int));
    bool needNewTopo = (loops != loopStore) || (cHashChk != cHash) || (fHashChk != fHash);
    if (needNewTopo) {


        // Convert the Maya faces to std::vectors
        std::vector<size_t> stdFaces;
        stdFaces.reserve(faces.length());
        for (auto f : faces) {
            stdFaces.push_back(f);
        }
        std::vector<size_t> stdCount;
        stdCount.reserve(count.length());
        for (auto c : count) {
            stdCount.push_back(c);
        }

        // Get UV faces and copy to std::vectors
        MIntArray uvCount, uvFaces;
        inFnMesh.getAssignedUVs(uvCount, uvFaces);

        std::vector<size_t> stdUvCount;
        stdUvCount.reserve(uvCount.length());
        for (auto c : uvCount) {
            stdUvCount.push_back(c);
        }

        std::vector<size_t> stdUvFaces;
        stdUvFaces.reserve(uvFaces.length());
        for (auto f : uvFaces) {
            stdUvFaces.push_back(f);
        }


        // Maya vs Max algorithm
        // The winding is backward in Max, but I want to keep the algorithm
        // consistent, so I pre/post reverse the faces
        stdFaces = reverseFaces(stdCount, stdFaces, 0);
        stdUvFaces = reverseFaces(stdUvCount, stdUvFaces, 0);

        auto faceVertIdxBorderPairs = findFaceVertBorderPairs(stdCount, stdFaces);
        auto [stdCountOut, stdFacesOut, retGrid, bVerts] = shellTopo(faceVertIdxBorderPairs, stdFaces, stdCount, inFnMesh.numVertices(), loops);
        auto [stdUvFacesOut, stdUvCountOut, uvGrid, uvEdges, bUvIdxs] = shellUvTopo(faceVertIdxBorderPairs, stdUvFaces, stdUvCount, inFnMesh.numUVs(), loops);
        grid = retGrid;

        stdFacesOut = reverseFaces(stdCountOut, stdFacesOut, 0);
        stdUvFacesOut = reverseFaces(stdUvCountOut, stdUvFacesOut, 0);




        // Build the output maya arrays for the faces
        newCount.setLength((uint)stdCountOut.size());
        for (size_t i = 0; i < stdCountOut.size(); ++i) {
            newCount[(uint)i] = (int)stdCountOut[i];
        }
        newFaces.setLength((uint)stdFacesOut.size());
        for (size_t i = 0; i < stdFacesOut.size(); ++i) {
            newFaces[(uint)i] = (int)stdFacesOut[i];
        }

        // Build the output maya arrays for the UVs
        newUvCount.setLength((uint)stdUvCountOut.size());
        for (size_t i = 0; i < stdUvCountOut.size(); ++i) {
            newUvCount[(uint)i] = (int)stdUvCountOut[i];
        }
        newUvFaces.setLength((uint)stdUvFacesOut.size());
        for (size_t i = 0; i < stdUvFacesOut.size(); ++i) {
            newUvFaces[(uint)i] = (int)stdUvFacesOut[i];
        }




        // Calculate the static UV positions
        MFloatArray uArray, vArray;
        inFnMesh.getUVs(uArray, vArray);
        std::vector<float> uvs;
        uvs.reserve((size_t)uArray.length() * 2);

        for (uint i = 0; i < uArray.length(); ++i) {
            uvs.push_back(uArray[i]);
            uvs.push_back(vArray[i]);
        }

        std::vector<float> newUvs = shellUvGridPos(uvs, uvGrid, uvEdges, uvOffset);
        newUArray.setLength((uint)newUvs.size() / 2);
        newVArray.setLength((uint)newUvs.size() / 2);
        for (uint i = 0; i < newUvs.size() / 2; i++) {
                newUArray[i] = newUvs[2 * (size_t)i];
                newVArray[i] = newUvs[2 * (size_t)i + 1];
        }



    }

    // Always calculate the dynamic vertex point positions
    MFloatPointArray verts, newVerts;
    MFloatVectorArray normals;
    inFnMesh.getPoints(verts);
    inFnMesh.getVertexNormals(false, normals);
    newVerts = shellVertGridPosMaya(verts, normals, grid, neg, pos);

	MDataHandle hOutput = dataBlock.outputValue(aOutputGeom);

    if (needNewTopo) {
        // Create a new mesh MObject
        MFnMeshData meshCreator;
        outMesh = meshCreator.create();
        MFnMesh meshFn;
        meshFn.create(newVerts.length(), newCount.length(), newVerts, newCount, newFaces, outMesh);
        meshFn.setUVs(newUArray, newVArray);
        meshFn.assignUVs(newUvCount, newUvFaces);

        hOutput.set(outMesh);
    }
    else {
        hOutput.set(outMesh);
        MFnMesh meshFn(outMesh);
        meshFn.setPoints(newVerts);
    }

    hOutput.setClean();
    return status;
}
