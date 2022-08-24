import numpy as np


def _lens(vecs):
    """Get the lengths of a bunch of vectors
    Use the einsum instead of the dot product because its
    it's almost always faster than np.dot
    """
    return np.sqrt(np.einsum('...ij,...ij->...i', vecs, vecs))


def _norm(vecs):
    """Normalize an array of vectors"""
    return vecs / _lens(vecs)[..., None]


def buildNormals(anim, counts, faces):
    """Build the per-vertex normals for a mesh

    Arguments:
        anim (np.array): A (frames, vertIdx, component) array of vertex positions
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: A (frames, vertIdx, component) array of per-vertex normals
    """
    # Get all adjacent vertex triples on the mesh
    nextStep = np.arange(len(faces)) + 1

    ranges = np.r_[0, counts.cumsum()]
    ends = ranges[1:] - 1
    nextStep[ends] = ranges[:-1]

    trips = np.empty((len(faces), 3), dtype=int)
    step = faces[nextStep]
    trips[:, 0] = faces
    trips[:, 1] = step
    trips[:, 2] = step[nextStep]

    # Get each edge as animated vectors
    anim = anim.swapaxes(0, 1)
    v1 = anim[trips[:, 0]] - anim[trips[:, 1]]
    v2 = anim[trips[:, 2]] - anim[trips[:, 1]]

    # Possibly get the triangle areas for weighting
    # Technically I should divide by 2, but since the ratio is what matters
    # I don't actually need to do that

    # 3dsMax doesn't take the triangle areas into account when building the
    # normals, but I could if I wanted to
    # triAreas = _lens(np.cross(v1, v2))

    # TODO: Handle zero vectors

    # Normalize the vectors
    v1 = _norm(v1)
    v2 = _norm(v2)

    # expand normalizing the face-vert Normals
    # because we can use the crossProd length to determine
    # the angle between the edges
    fvn = np.cross(v2, v1)
    fvnLens = _lens(fvn)
    fvn = fvn / fvnLens[..., None]
    angles = np.arcsin(fvnLens)

    # Weight by the angles
    weighted = fvn * angles[..., None]

    # Now use a ufunc to sum things based on the center vert
    out = np.zeros(anim.shape)
    np.add.at(out, trips[:, 1], weighted)
    out = _norm(out)
    out[np.isnan(out)] = 0.0
    out = out.swapaxes(0, 1)
    return out


def triangulateFaces(counts, faces):
    """Given the counts and faces of a mesh, return
    a faces array of the triangulated mesh

    Arguments:
        counts (np.array): The number of vertices per face
        faces (np.array): The indices of vertices per face, flattened

    Returns:
        np.array: An N*3 array of vertex indices that will build a fully
            triangulated mesh
        np.array: An N*3 array of indices into the faces array saying which
            face-vert became this triangle
        np.array: An array of face indices that say which polygon each
            triangle was built from
    """
    # Get the biggest polygon size
    maxC = counts.max()

    # Build a single polygon of that size
    # So like, if the polygon was size 5
    # Build the triangle fan: [0,1,2,  0,2,3,  0,3,4]
    remap = np.zeros(maxC + (maxC - 3) * 2, dtype=int)
    rng = np.arange(len(remap) / 3, dtype=int)
    remap[1::3] = rng + 1
    remap[2::3] = rng + 2

    # build an array of the number of vertices that will be
    # in each fan
    recounts = counts + (counts - 3) * 2

    # Build the indices into the original faces array
    # based on the remap and recounts variables

    # get the pairwise start/end of each fan range
    rccs = np.r_[0, recounts.cumsum()]
    # offset each fan by the cumulative number of vertices
    smear = np.arange(rccs[-1]) - rccs[:-1].repeat(recounts)
    # Get the pairwise start/end of each polygon range
    repc = np.r_[0, counts.cumsum()][:-1]
    # Add the repeated polygon offsets to the fan offsets
    # to get my final indexing into the faces array
    fvIdxs = remap[smear] + repc.repeat(recounts)
    fvIdxs = fvIdxs.reshape((-1, 3))
    # Build the final triangle output
    tris = faces[fvIdxs]

    # also get the face that each tri comes from
    faceIdxs = np.repeat(np.arange(tris.size), counts - 2)

    return tris, fvIdxs, faceIdxs


def reverseFaces(counts, faces):
    """Reverse the winding of the given faces

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: The new reversed flattened indices of each vert
    """
    # Build idAry to be "the number of steps left/right to get the next index"
    # So if everything was -1, that would mean "grab the previous item in the list"
    # And that works for almost everything, except when we want the next face chunk
    # Then we've gotta step from the beginning of one face, to the end of the next,
    # That means we step the sum of the two adjacent faces (minus 1)
    # Then we make sure to start at the final index of the first face
    # Then just apply that rule to jumble the values of "faces" for the output

    # Start with all -1's
    idAry = np.ones(len(faces), dtype=int) * -1
    # Get the indices of the last index of each face
    ends = np.r_[0, counts[:-1].cumsum()]
    # Get the 2-face-step sizes
    idAry[ends[1:]] = counts[1:] + counts[:-1] - 1
    # Start at the last index of the first face
    idAry[ends[0]] = counts[0] - 1
    # Apply the rules for the output
    return faces[idAry.cumsum()]


def getEdgePairIdxs(counts, faces):
    """ Build a 2*N array of ordered edge pairs """
    # We're going to build a list of how many steps to take to get
    # the next face-vertex. Most of those values will be 1 (meaning
    # one step to the right), but at the end of each face, we need
    # to step left back to the beginning

    # Start with all 1's
    idAry = np.ones(len(faces), dtype=int)
    # Get the indices of the last index of each face
    ends = np.r_[0, counts[:-1].cumsum()]
    # Get how many steps back it is to the beginning of the face
    idAry[ends] = -counts
    # apply that offset to an arange to get the index of the next face-vert

    # Now add the step value to the index value of each spot in the array
    # to get the real indices of the next value around each face
    return np.arange(len(faces), dtype=int) + idAry


def findInArray(needles, haystack):
    """Search for the indices of each needle in the haystack"""
    # Much faster than using dicts
    sorter = np.argsort(haystack)
    return sorter[np.searchsorted(haystack, needles, sorter=sorter)]


def buildCycles2(idxs, edges):
    """Build cycles with the given indices and edges"""
    return findInArray(edges[:, 1], idxs)

def buildCycles(edges):
    """Build cycles with the given indices and edges"""
    return findInArray(edges[:, 1], edges[:, 0])




def findFaceVertBorderPairs(counts, faces):
    """ Find all border edges and their face indices
    But make sure they're in the order that 3dsMax would provide them

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: An N*2 array of properly ordered vertex pairs
        np.array: An N*2 array of indices into the faces array to
            say which face-verts became these indices
        np.array: An array of length N of the face index that
            each vertex pair originated from
    """
    nextIdxs = getEdgePairIdxs(counts, faces)
    edgeDict = {}
    for eIdx, nIdx in enumerate(nextIdxs):
        a = faces[eIdx]
        b = faces[nIdx]
        key = (a, b) if a < b else (b, a)
        if key in edgeDict:
            del edgeDict[key]
        else:
            edgeDict[key] = eIdx

    startIdxs = np.array(sorted(edgeDict.values()))
    endIdxs = nextIdxs[startIdxs]

    faceVertIdxBorderPairs = np.stack((startIdxs, endIdxs), axis=1)
    return faceVertIdxBorderPairs



def findSortedBorderEdges(counts, faces):
    faceVertIdxBorderPairs = findFaceVertBorderPairs(counts, faces)
    edges = faces[faceVertIdxBorderPairs]
    stops = np.setdiff1d(edges[:, 1], edges[:, 0], assume_unique=True)











def buildCycleBridges(startBorderVerts, endBorderVerts, cycle, bridgeFirstIdx, numBridgeSegs):
    """ Build a bridge between a starting set of edges, and an ending set of edges
    We assume that the starting/ending edge sets have the same order
    We also assume that there are no edges along the border that don't form full
    cycles

    Arguments:
        startBorderVerts (np.array): A Flat array of vertex indices that define
            border edges for the starting loop
        endBorderVerts (np.array): A Flat array of vertex indices that define
            border edges for the ending loop
        cycle (np.array): An array that defines the index of the vertex that's
            the clockwise neighbor of the given index
        bridgeFirstIdx (int): Where to start counting for newly created bridge indices
        numBridgeSegs (int): The number of segments that the bridge will have between
            the start and end. It has a minimum of 1

    Returns:
        np.array: The Face array that was created. This should extend the existing
            face array for whatever mesh you're working on
        np.array: The Count array that was created. This should extend the existing
            count array for whatever mesh you're working on
    """
    numBorders = len(startBorderVerts)

    oge, ogs = np.ogrid[:numBorders, :numBridgeSegs]
    bFaces = np.zeros((numBorders, numBridgeSegs, 4), dtype=int)

    # Build the raw bridge indices
    # For simplicity of code, I don't special-case the parts where
    # I'm going to connect back to the original meshes
    bFaces[:, :, 0] = oge + (numBorders * (ogs - 1))
    bFaces[:, :, 1] = cycle[oge] + (numBorders * (ogs - 1))
    bFaces[:, :, 2] = cycle[oge] + (numBorders * ogs)
    bFaces[:, :, 3] = oge + (numBorders * ogs)

    # Offset these indices
    bFaces += bridgeFirstIdx

    # Connect them back to the original meshes
    bFaces[:, 0, 0] = startBorderVerts[oge].flatten()
    bFaces[:, 0, 1] = startBorderVerts[cycle[oge]].flatten()
    bFaces[:, -1, 2] = endBorderVerts[cycle[oge]].flatten()
    bFaces[:, -1, 3] = endBorderVerts[oge].flatten()
    bFaces = bFaces.flatten()

    # Counts is just a bunch of 4's
    bCounts = np.full((numBorders * numBridgeSegs), 4)

    return bFaces, bCounts
















def getBridgeIdx(eIdx, segIdx, numBridgeSegs, vertCount, bIdxs):
    """ This is how we get the bridge indices in c++ """
    if segIdx == 0:
        return bIdxs[eIdx]
    if segIdx == numBridgeSegs:
        return bIdxs[eIdx] + vertCount
    return (2 * vertCount) + eIdx + (len(bIdxs) * (segIdx - 1))




def angleBetweenVectors(a, b):
    # Apparently MUCH more numerically stable than
    # acos of the dot product of the norms
    an = a.norm()
    bn = b.norm()
    return 2 * np.atan2((an + bn).length(), (an - bn).length())



def heightOfMidEdge(a, b, offset):
    an = a.norm()
    bn = b.norm()
    return offset * (2 / (1 - np.dot(an, bn)))**0.5


def shellUVs(vertFaces, oUvFaces, oUvCounts, numUVs, numBridgeSegs):
    """
    vertFaces (np.ndarray): The numpy array of faces for vertex indices
    oUvFaces (np.ndarray): The numpy array of faces for UV indices

    """




    # distance of diagonal extension in UV space: H
    # edge ofset: E


    # cos of angle between a and b: CosAng = dot(a.norm(), b.norm()))
    # sin of angle between a and b: SinAng = determinant(a.norm(), b.norm()))
    # real counterclockwise angle = atan2(cosAng/sinAng)
    # H = E * sqrt(2 / (1 - CosAng))

    vertBorderPairs, faceVertIdxBorderPairs = findSortedBorderEdges(oUvCounts, oUvFaces)
    uvBorderPairs = oUvFaces[faceVertIdxBorderPairs]


    startMap = {uvBorderPairs[i][0]: i for i in range(len(uvBorderPairs))}
    forwardCycles = np.array([startMap.get(edge[1], np.nan) for edge in uvBorderPairs])

    endMap = {uvBorderPairs[i][1]: i for i in range(len(uvBorderPairs))}
    backwardCycles = np.array([endMap.get(edge[0], np.nan) for edge in uvBorderPairs])












    return forwardCycles, backwardCycles









def findSortedBorderEdges(counts, faces):
    """ Find all border edges and their face indices
    But make sure they're in the order that 3dsMax would provide them

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: An N*2 array of properly ordered vertex pairs
        np.array: An N*2 array of indices into the faces array to
            say which face-verts became these indices
        np.array: An array of length N of the face index that
            each vertex pair originated from
    """
    nextIdxs = getEdgePairIdxs(counts, faces)
    edgeDict = {}
    for eIdx, nIdx in enumerate(nextIdxs):
        a = faces[eIdx]
        b = faces[nIdx]
        key = (a, b) if a < b else (b, a)
        if key in edgeDict:
            del edgeDict[key]
        else:
            edgeDict[key] = eIdx

    startIdxs = np.array(sorted(edgeDict.values()))
    endIdxs = nextIdxs[startIdxs]

    faceVertIdxBorderPairs = np.stack((startIdxs, endIdxs), axis=1)
    borderPairs = faces[faceVertIdxBorderPairs]
    return borderPairs, faceVertIdxBorderPairs



def shellTopo(oFaces, oCounts, vertCount, numBridgeSegs):
    origBorderEdges, faceVertIdxBorderPairs = findSortedBorderEdges(oCounts, oFaces)
    cycle = buildCycles(origBorderEdges, origBorderEdges[:, 0])

    borderVerts = [e[0] for e in origBorderEdges]

    brEdges = np.full(numBridgeSegs * len(borderVerts), 4, dtype=int)
    counts = np.concatenate((oCounts, oCounts, brEdges), axis=0)

    numBorders = len(borderVerts)
    offset = 2 * vertCount

    oge, ogs = np.ogrid[:numBorders, :numBridgeSegs]
    bFaces = np.zeros((numBorders, numBridgeSegs, 4), dtype=int)

    # Build the bridge indices
    bFaces[:, :, 0] = offset + oge + (numBorders * (ogs - 1))
    bFaces[:, :, 1] = offset + cycle[oge] + (numBorders * (ogs - 1))
    bFaces[:, :, 2] = offset + cycle[oge] + (numBorders * ogs)
    bFaces[:, :, 3] = offset + oge + (numBorders * ogs)

    # Connect them back to the original meshes
    bFaces[:, 0, 0] = borderVerts[oge].flatten()
    bFaces[:, 0, 1] = borderVerts[cycle[oge]].flatten()
    bFaces[:, -1, 2] = borderVerts[cycle[oge]].flatten() + vertCount
    bFaces[:, -1, 3] = borderVerts[oge].flatten() + vertCount

    iFaces = reverseFaces(oCounts, oFaces) + vertCount
    faces = np.concatenate((oFaces, iFaces, bFaces))

    return counts, faces, borderVerts







def shellGeo(rawAnim, normals, bIdxs, numBridgeSegs, innerOffset, outerOffset):
    numFrames = len(rawAnim)

    bVertCount = (numBridgeSegs - 1) * bIdxs.length()
    bVerts = np.zeros((numFrames, bVertCount, 3))
    ret = np.concatenate(
        (
            rawAnim + normals * outerOffset,
            rawAnim - normals * innerOffset,
            bVerts,
        )
    )

    innerVerts = ret[:, bIdxs]
    outerVerts = ret[:, bIdxs + len(rawAnim)]
    for segIdx in range(1, numBridgeSegs):
        perc = float(segIdx) / numBridgeSegs
        ring = innerVerts * (1.0 - perc) + outerVerts * perc
        ringIdxs = bIdxs + ((2 * len(rawAnim)) + (len(bIdxs) * (segIdx - 1)))
        ret[:, ringIdxs] = ring

    return ret


def shell(anim, faces, counts, innerOffset, outerOffset, bridgeSegs):
    vertCount = len(anim[1])
    sCounts, sFaces, borderIdxs = shellTopo(faces, counts, vertCount, bridgeSegs)
    normals = buildNormals(anim, counts, faces)
    sAnim = shellGeo(anim, normals, borderIdxs, bridgeSegs, innerOffset, outerOffset)

    return sAnim, sFaces, sCounts

