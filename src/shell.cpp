#include <maya/MTypeId.h> 

#include <maya/MTime.h> 
#include <maya/MMatrix.h>
#include <maya/MDoubleArray.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MTransformationMatrix.h>

#include <maya/MItGeometry.h>
#include <maya/MItMeshVertex.h>
#include <maya/MItSurfaceCV.h>

#include <maya/MFnNurbsSurface.h>
#include <maya/MFnMesh.h>

#include <maya/MFnUnitAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnEnumAttribute.h>

#include <vector>
#include <pair>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>  // std::sort
#include <numeric>    // std::iota

#include "shell.h"
#include "maxShellAlg.h"



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



    aPosThickness = fnUnit.create("posThickness", "pt", MFnUnitAttribute::kDistance, 0.0);
    fnUnit.setStorable(true);
    fnUnit.setKeyable(true);
    stat = addAttribute(aPosThickness);

    aPosThickness = fnUnit.create("negThickness", "nt", MFnUnitAttribute::kDistance, 0.0);
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
    stat = addAttribute(aInputGeom);


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
	MPlug& plug,
	MDataBlock& dataBlock
){
	MStatus status;


    if (plug != aOutputGeom){
        // Only the output geom plug matters
        return MStatus::kSuccess;
    }

	MDataHandle hPos = dataBlock.inputValue(aPosThickness);
	float pos = hPos.asFloat();

	MDataHandle hNeg = dataBlock.inputValue(aNegThickness);
	float neg = hNeg.asFloat();

	MDataHandle hLoops = dataBlock.inputValue(aThickLoops);
	int loops = hLoops.asInt();

	MDataHandle hInput = dataBlock.inputValue(aInputGeom);
	MDataHandle hOutput = dataBlock.outputValue(aOutputGeom);
    hOutput.set(hInput.asMesh());
    MObject outMesh = hOutput.asMesh();

    MFnMesh meshFn(outMesh);

    // Do the topology
    MIntArray count, faces;
    meshFn.getVertices(count, faces);
    std::vector<int> retFaces, retCounts;
    auto bVerts = shellTopo(faces, counts, meshFn.numVertices(), loops, retFaces, retCounts);
    MIntArray newCount(retCounts.data()), newFaces(retFaces.data());

    // Do the point positions
    MPointArray verts;
    MFloatVectorArray normals;
    meshFn.getPoints(verts);
    meshFn.getVertexNormals(false, normals);
    auto newVerts = shellGeo(verts, normals, bVerts, loops, neg, pos);

    // Make the output
    meshFn.createInPlace(newVerts.length(), newCount.length(), newVerts, newCount, newFaces);
    // TODO: Handle UV's
    return status;
}







MStatus shell::deform(
	MDataBlock& dataBlock,
	MItGeometry& geoIter,
	const MMatrix& wmat,
	unsigned int multiIndex
){
	MStatus status;

	MDataHandle hEnv = dataBlock.inputValue(envelope);
	float env = hEnv.asFloat();
    if (env == 0.0) {
        return MStatus::kSuccess;
    }

    // Get the time input
    MDataHandle hTime = dataBlock.inputValue(aTime);
    float time = (float)hTime.asTime().asUnits(MTime::kSeconds);


    // Get the octave loop data
    MDataHandle hOctaveCount = dataBlock.inputValue(aOctaveCount);
    MDataHandle hOctaveScaleBase = dataBlock.inputValue(aOctaveScaleBase);
    MDataHandle hOctaveRangeBase = dataBlock.inputValue(aOctaveRangeBase);
    int octaveCount = hOctaveCount.asInt();
    double octaveScaleBase = 1.0 / hOctaveScaleBase.asDouble();
    double octaveRangeBase = 1.0 / hOctaveRangeBase.asDouble();
    //if (octaveCount > 1) {fnFractal->SetOctaveCount(octaveCount); }

    // Get the amplitude data
    MDataHandle hAmp = dataBlock.inputValue(aAmp);
    MDataHandle hAmpOffset = dataBlock.inputValue(aAmpOffset);
    MDataHandle hAmpClampHigh = dataBlock.inputValue(aAmpClampHigh);
    MDataHandle hAmpClampLow = dataBlock.inputValue(aAmpClampLow);
    MDataHandle hAmpUseClampHigh = dataBlock.inputValue(aAmpUseClampHigh);
    MDataHandle hAmpUseClampLow = dataBlock.inputValue(aAmpUseClampLow);
    double amp = hAmp.asDouble();
    double ampOffset = hAmpOffset.asDouble();
    double ampClampHigh = hAmpClampHigh.asDouble();
    double ampClampLow = hAmpClampLow.asDouble();
    bool ampUseClampHigh = hAmpUseClampHigh.asBool();
    bool ampUseClampLow = hAmpUseClampLow.asBool();


    // Get the SRTSh inputs and compose them into a matrix
    // I should probably set the shear to be hidden
    MDataHandle hTranslate = dataBlock.inputValue(aTranslate);
    MDataHandle hRotate = dataBlock.inputValue(aRotate);
    MDataHandle hScale = dataBlock.inputValue(aScale);
    MDataHandle hShear = dataBlock.inputValue(aShear);
    MDataHandle hRotateOrder = dataBlock.inputValue(aRotateOrder);
    MVector translate = hTranslate.asVector();
    double *rotate = hRotate.asDouble3();
    double *scale = hScale.asDouble3();
    double *shear = hShear.asDouble3();
    short rotateOrder = hRotateOrder.asShort();
    MTransformationMatrix::RotationOrder rotationOrders[6] = {
        MTransformationMatrix::kXYZ, MTransformationMatrix::kYZX,
        MTransformationMatrix::kZXY, MTransformationMatrix::kXZY,
        MTransformationMatrix::kYXZ, MTransformationMatrix::kZYX
    };
    MTransformationMatrix mtm;
    mtm.setScale(scale, MSpace::kWorld);
    mtm.setRotation(rotate, rotationOrders[rotateOrder]);
    mtm.setTranslation(translate, MSpace::kWorld);
    mtm.setShear(shear, MSpace::kWorld);
    MMatrix mat = mtm.asMatrixInverse();

	// Maya automatically copies the input plug to the output plug
	// and then gives you an iterator over that
	// So get the OUTPUT handle for this mutiIndex
	MPlug outPlug(thisMObject(), outputGeom);
	outPlug.selectAncestorLogicalIndex(multiIndex, outputGeom);
	MDataHandle hOutput = dataBlock.outputValue(outPlug);


    MPointArray worldPts, localPts, newPts;
    geoIter.allPositions(worldPts, MSpace::kWorld);
    geoIter.allPositions(localPts, MSpace::kObject);
    newPts.setLength(worldPts.length());

    // store the indices of all the geo we're looping over
    std::vector<int> allIdxs(geoIter.count());
    std::vector<float> allWeights(geoIter.count());
    for (; !geoIter.isDone(); geoIter.next()) {
        int pi = geoIter.positionIndex();
        int idx = geoIter.index();
        allIdxs[pi] = idx;
        allWeights[pi] = weightValue(dataBlock, multiIndex, idx);
    }
    geoIter.reset();


    // Get the per-point normals in object space
    MFloatVectorArray allNorms;
    allNorms.setLength(geoIter.count());
    auto outType = hOutput.type();
    if (outType == MFnData::kMesh) {
        MObject mesh = hOutput.asMesh();
        MFnMesh meshFn(mesh);
        MFloatVectorArray tnorms;
        meshFn.getVertexNormals(false, tnorms);
        // match the norms array to the allpts array
        int i = 0;
        for (const int &idx: allIdxs){
            allNorms[i++] = tnorms[idx];
        }
    }
    else if (outType == MFnData::kNurbsSurface) {
        MObject surface = hOutput.asNurbsSurface();
        MFnNurbsSurface fnSurf(surface);
        MDoubleArray uParams, vParams;
        getSurfaceCVParams(fnSurf, uParams, vParams);
        int i = 0;
        // It's possible to do this without the tnorms
        // but this works well enough in practice
        MFloatVectorArray tnorms;
        for (const double &u: uParams){
            for (const double &v: vParams){
                tnorms[i++] = fnSurf.normal(u, v, MSpace::kObject);
            }
        }
        int idx = 0;
        for (const int &idx: allIdxs){
            allNorms[i++] = tnorms[idx];
        }
    }
    else {
        // Only mesh and nurbs supported
        return MStatus::kFailure;
    }

    // Copy the world data from the array-of-arrays mpoint array 
    // into the multiple flat vectors, then run all the noise
    // at once in parallel

    std::vector<float> noiseVals(worldPts.length());
    float scaleBase = 1.0;
    float rangeBase = 1.0;
    for (int octave = 0; octave < octaveCount; ++octave) {

        #pragma omp parallel for
        for (int i=0; i<(int)worldPts.length(); ++i){
            MPoint pos = worldPts[i] * mat;
            float n = (float)noise4_XYZBeforeW(ose, osg, pos.x*scaleBase, pos.y*scaleBase, pos.z*scaleBase, time);
            noiseVals[i] += n / rangeBase;
        }

        scaleBase *= (float)octaveScaleBase;
        rangeBase *= (float)octaveRangeBase;
    }


    // Move the points based off the noise value
    //# pragma omp parallel for
    for (int idx=0; idx<allIdxs.size(); ++idx){
        float weight = allWeights[idx];
        if (weight == 0.0) {
            newPts[idx] = localPts[idx];
            continue;
        }

        double n = ((double)noiseVals[idx] * amp) + ampOffset;
        if (ampUseClampHigh && n > ampClampHigh)
            n = ampClampHigh;
        if (ampUseClampLow && n < ampClampLow)
            n = ampClampLow;

        newPts[idx] = localPts[idx] + (allNorms[idx] * weight * n * env);           
    }

    geoIter.setAllPositions(newPts, MSpace::kObject);
    return MStatus::kSuccess;
}
