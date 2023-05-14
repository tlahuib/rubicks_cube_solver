package main

import (
	"fmt"
)

type Cube struct {
	// Cube     [6][3][3]rune
	Pieces   [3][3][3]Piece
	IsSolved bool
}
type Piece struct {
	Colors   []rune
	Location [3]int
	Rotation []int // Same shape as colors. Each color references a face (0, 5)
}
type Move struct {
	Axis      int
	Line      int
	Direction bool
}

var solvedCube = [][][][]rune{
	{
		{{'w', 'g', 'r'}, {'w', 'r'}, {'w', 'b', 'r'}},
		{{'w', 'g'}, {'w'}, {'w', 'b'}},
		{{'w', 'g', 'o'}, {'w', 'o'}, {'w', 'b', 'o'}},
	}, {
		{{'g', 'r'}, {'r'}, {'b', 'r'}},
		{{'g'}, {}, {'b'}},
		{{'g', 'o'}, {'o'}, {'b', 'o'}},
	}, {
		{{'y', 'g', 'r'}, {'y', 'r'}, {'y', 'b', 'r'}},
		{{'y', 'g'}, {'y'}, {'y', 'b'}},
		{{'y', 'g', 'o'}, {'y', 'o'}, {'y', 'b', 'o'}},
	},
}
var faceColors = map[rune]int{
	'w': 0,
	'y': 1,
	'g': 2,
	'b': 3,
	'r': 4,
	'o': 5,
}
var pieceToFace = map[[3]int]map[int][2]int{
	{0, 0, 0}: {0: {0, 0}, 2: {0, 2}, 4: {2, 0}},
	{0, 0, 1}: {0: {0, 1}, 4: {2, 1}},
	{0, 0, 2}: {0: {0, 2}, 3: {0, 0}, 4: {2, 2}},
	{0, 1, 0}: {0: {1, 0}, 2: {1, 2}},
	{0, 1, 1}: {0: {1, 1}},
	{0, 1, 2}: {0: {1, 2}, 3: {1, 0}},
	{0, 2, 0}: {0: {2, 0}, 2: {2, 2}, 5: {0, 0}},
	{0, 2, 1}: {0: {2, 1}, 5: {0, 1}},
	{0, 2, 2}: {0: {2, 2}, 3: {2, 0}, 5: {0, 2}},
	{1, 0, 0}: {2: {0, 1}, 4: {1, 0}},
	{1, 0, 1}: {4: {1, 1}},
	{1, 0, 2}: {3: {0, 1}, 4: {1, 2}},
	{1, 1, 0}: {2: {1, 1}},
	{1, 1, 1}: {},
	{1, 1, 2}: {3: {1, 1}},
	{1, 2, 0}: {2: {2, 1}, 5: {1, 0}},
	{1, 2, 1}: {5: {1, 1}},
	{1, 2, 2}: {3: {2, 1}, 5: {1, 2}},
	{2, 0, 0}: {2: {0, 0}, 4: {0, 0}, 1: {0, 2}},
	{2, 0, 1}: {4: {0, 1}, 1: {0, 1}},
	{2, 0, 2}: {1: {0, 0}, 3: {0, 2}, 4: {0, 2}},
	{2, 1, 0}: {2: {1, 0}, 1: {1, 2}},
	{2, 1, 1}: {1: {1, 1}},
	{2, 1, 2}: {1: {1, 0}, 3: {1, 2}},
	{2, 2, 0}: {2: {2, 0}, 1: {2, 2}, 5: {2, 0}},
	{2, 2, 1}: {5: {2, 1}, 1: {2, 1}},
	{2, 2, 2}: {1: {2, 0}, 3: {2, 2}, 5: {2, 2}},
}
var blank string = "         "

// var AxisRelation = [3][4]int8{
// 	{0, 3, 1, 2}, // Axis x, moves front, right, back and left faces
// 	{0, 4, 1, 5}, // Axis y, moves front, top, back and bottom faces
// 	{2, 4, 3, 5}, // Axis z, moves left, top, right and bottom faces
// }
// var MoveNotation = map[Move]string{
// 	{Axis: 0, Line: 0, Direction: false}: "U",
// 	{Axis: 0, Line: 0, Direction: true}:  "U'",
// 	{Axis: 0, Line: 1, Direction: false}: "E'",
// 	{Axis: 0, Line: 1, Direction: true}:  "E",
// 	{Axis: 0, Line: 2, Direction: false}: "D'",
// 	{Axis: 0, Line: 2, Direction: true}:  "D",
// 	{Axis: 1, Line: 0, Direction: false}: "L",
// 	{Axis: 1, Line: 0, Direction: true}:  "L'",
// 	{Axis: 1, Line: 1, Direction: false}: "M",
// 	{Axis: 1, Line: 1, Direction: true}:  "M'",
// 	{Axis: 1, Line: 2, Direction: false}: "R'",
// 	{Axis: 1, Line: 2, Direction: true}:  "R",
// 	{Axis: 2, Line: 0, Direction: false}: "F'",
// 	{Axis: 2, Line: 0, Direction: true}:  "F",
// 	{Axis: 2, Line: 1, Direction: false}: "S'[layer][row][col]",
// 	{Axis: 2, Line: 1, Direction: true}:  "S",
// 	{Axis: 2, Line: 2, Direction: false}: "B",
// 	{Axis: 2, Line: 2, Direction: true}:  "B'",
// }
// var faceDistances = map[[2]int]int{
// 	{0, 0}: 0, {0, 1}: 2, {0, 2}: -1, {0, 3}: 1, {0, 4}: 1, {0, 5}: -1,
// 	{1, 0}: -2, {1, 1}: 0, {1, 2}: 1, {1, 3}: -1, {1, 4}: -1, {1, 5}: 1,
// 	{2, 0}: 1, {2, 1}: -1, {2, 2}: 0, {2, 3}: 2, {2, 4}: 1, {2, 5}: -1,
// 	{3, 0}: -1, {3, 1}: 1, {3, 2}: -2, {3, 3}: 0, {3, 4}: -1, {3, 5}: 1,
// 	{4, 0}: -1, {4, 1}: 1, {4, 2}: -1, {4, 3}: 1, {4, 4}: 0, {4, 5}: 2,
// 	{5, 0}: 1, {5, 1}: -1, {5, 2}: 1, {5, 3}: -1, {5, 4}: -2, {5, 5}: 0,
// }

func getInitialRotations(colors []rune) []int {
	var rotations []int

	for _, color := range colors {
		rotations = append(rotations, faceColors[color])
	}

	return rotations
}

func initializeCube() Cube {
	// The array is face, line, column
	// The faces are ordered front, back, left, right, top, bottom
	var cube Cube

	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				colors := solvedCube[layer][row][col]
				location := [3]int{layer, row, col}
				rotation := getInitialRotations(colors)
				cube.Pieces[layer][row][col] = Piece{Colors: colors, Location: location, Rotation: rotation}
			}
		}
	}
	cube.IsSolved = true

	return cube
}

// func rotateFace(cube Cube, face int, direction bool) Cube {
// 	_cube := cube
// 	for i := 0; i < 3; i++ {
// 		for j := 0; j < 3; j++ {
// 			if direction {
// 				_cube.Cube[face][j][2-i] = cube.Cube[face][i][j]
// 			} else {
// 				_cube.Cube[face][2-j][i] = cube.Cube[face][i][j]
// 			}
// 		}
// 	}
// 	return _cube
// }

// func reverseRows(face [3][3]rune) [3][3]rune {
// 	reversedRows := face

// 	reversedRows[0], reversedRows[2] = reversedRows[2], reversedRows[0]

// 	return reversedRows
// }

// func reverseCols(face [3][3]rune) [3][3]rune {
// 	reversedCols := face

// 	for i := 0; i < 3; i++ {
// 		reversedCols[i][0], reversedCols[i][2] = reversedCols[i][2], reversedCols[i][0]
// 	}

// 	return reversedCols
// }

// func reverseFace(face [3][3]rune) [3][3]rune {
// 	reversedFace := face

// 	reversedFace = reverseCols(reversedFace)
// 	reversedFace = reverseRows(reversedFace)

// 	return reversedFace
// }

// func rotateX(cube Cube, move Move) Cube {
// 	var ii [4]int
// 	var shift int
// 	_cube := cube

// 	// The direction changes loop and shift
// 	if move.Direction {
// 		ii = [4]int{3, 2, 1, 0}
// 		shift = 3
// 	} else {
// 		ii = [4]int{0, 1, 2, 3}
// 		shift = 1
// 	}

// 	// Iterate over rows// Face order is:
// 	//      |---|
// 	//      |-4-|
// 	//      |---|
// 	// |---||---||---||---|
// 	// |-2-||-0-||-3-||-1-|
// 	// |---||---||---||---|
// 	//      |---|
// 	//      |-5-|
// 	//      |---|
// 	for _, i := range ii {
// 		_cube.Cube[AxisRelation[0][i]][move.Line] = cube.Cube[AxisRelation[0][(i+shift)%4]][move.Line]
// 	}

// 	// Rotate top or bottom face
// 	switch move.Line {
// 	case 0:
// 		_cube = rotateFace(_cube, 4, !move.Direction)
// 	case 2:
// 		_cube = rotateFace(_cube, 5, move.Direction)
// 	}

// 	return _cube
// }

// func rotateY(cube Cube, move Move) Cube {
// 	var ii [4]int
// 	var shift int
// 	_cube := cube

// 	// Account for the back being reversed
// 	_cube.Cube[1] = reverseFace(_cube.Cube[1])
// 	referenceCube := _cube

// 	// The direction changes loop and shift
// 	if move.Direction {
// 		ii = [4]int{3, 2, 1, 0}
// 		shift = 3
// 	} else {
// 		ii = [4]int{0, 1, 2, 3}
// 		shift = 1
// 	}

// 	for _, i := range ii { // Iterate over faces
// 		for j := 0; j < 3; j++ { // Iterate over Rows
// 			_cube.Cube[AxisRelation[1][i]][j][move.Line] = referenceCube.Cube[AxisRelation[1][(i+shift)%4]][j][move.Line]
// 		}
// 	}

// 	// Undo the reversion
// 	_cube.Cube[1] = reverseFace(_cube.Cube[1])

// 	// Rotate left or right face// Output order is 'w', 'y', 'g', 'b', 'r', 'o'
// 	// Face order is:
// 	//      |---|
// 	//      |-4-|
// 	//      |---|
// 	// |---||---||---||---|
// 	// |-2-||-0-||-3-||-1-|
// 	// |---||---||---||---|
// 	//      |---|
// 	//      |-5-|
// 	//      |---|
// 	switch move.Line {
// 	case 0:
// 		_cube = rotateFace(_cube, 2, !move.Direction)
// 	case 2:
// 		_cube = rotateFace(_cube, 3, move.Direction)
// 	}

// 	return _cube
// }

// func rotateZ(cube Cube, move Move) Cube {
// 	var ii [4]int
// 	var shift int
// 	_cube := cube

// 	// Account for rows/columns reversion
// 	_cube.Cube[2] = reverseFace(_cube.Cube[2])
// 	_cube.Cube[4] = reverseRows(_cube.Cube[4])
// 	_cube.Cube[5] = reverseCols(_cube.Cube[5])
// 	referenceCube := _cube

// 	// The direction changes loop and shift
// 	if move.Direction {
// 		ii = [4]int{3, 2, 1, 0}
// 		shift = 3
// 	} else {
// 		ii = [4]int{0, 1, 2, 3}
// 		shift = 1
// 	}

// 	for _, i := range ii { // Iterate over faces
// 		if i%2 == 0 { // Left and right use columns, up and bottom use rows
// 			for j := 0; j < 3; j++ { // Iterate over elements
// 				_cube.Cube[AxisRelation[2][i]][j][move.Line] = referenceCube.Cube[AxisRelation[2][(i+shift)%4]][move.Line][j]
// 			}
// 		} else {
// 			for j := 0; j < 3; j++ { // Iterate over elements
// 				_cube.Cube[AxisRelation[2][i]][move.Line][j] = referenceCube.Cube[AxisRelation[2][(i+shift)%4]][j][move.Line]
// 			}
// 		}
// 	}

// 	// Undo the reversion
// 	_cube.Cube[2] = reverseFace(_cube.Cube[2])
// 	_cube.Cube[4] = reverseRows(_cube.Cube[4])
// 	_cube.Cube[5] = reverseCols(_cube.Cube[5])

// 	// Rotate front or back face
// 	switch move.Line {
// 	case 0:
// 		_cube = rotateFace(_cube, 0, move.Direction)
// 	case 2:
// 		_cube = rotateFace(_cube, 1, !move.Direction)
// 	}

// 	return _cube
// }

// func MoveCube(cube Cube, move Move) Cube {
// 	var _cube Cube

// 	switch move.Axis {
// 	case 0:
// 		_cube = rotateX(cube, move)
// 	case 1:
// 		_cube = rotateY(cube, move)
// 	case 2:
// 		_cube = rotateZ(cube, move)
// 	}

// 	_cube = CheckSolvedCube(_cube)
// 	return _cube
// }

// func CheckSolvedCube(cube Cube) Cube {
// 	_cube := cube
// 	for i := 0; i < 6; i++ {
// 		for j := 0; j < 3; j++ {
// 			for k := 0; k < 3; k++ {
// 				if cube.Cube[i][j][k] != cube.Cube[i][0][0] {
// 					_cube.IsSolved = false
// 					return _cube
// 				}
// 			}
// 		}
// 	}

// 	_cube.IsSolved = true
// 	return _cube
// }

// func scrambleCube(cube Cube, nMoves int) (Cube, []string) {
// 	var moves []string
// 	_cube := cube

// 	for i := 0; i < nMoves; i++ {
// 		move := Move{rand.Intn(3), rand.Intn(3), rand.Float32() <= 0.5}
// 		moves = append(moves, MoveNotation[move])
// 		_cube = MoveCube(_cube, move)
// 	}
// 	return _cube, moves
// }

// func InitializeScrambledCube(nMoves int) Cube {
// 	cube := initializeCube()

// 	cube, _ = scrambleCube(cube, nMoves)

// 	return cube
// }

func stringifyLine(line [3]rune) string {
	var strLine string
	for i := 0; i < 3; i++ {
		strLine += "|" + string(line[i]) + "|"
	}
	return strLine
}

func CubeToFaces(cube Cube) [6][3][3]rune {
	var faces [6][3][3]rune

	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				piece := cube.Pieces[layer][row][col]
				faceMap := pieceToFace[piece.Location]
				for i, faceColor := range piece.Colors {
					face := piece.Rotation[i]
					coord := faceMap[face]
					faces[face][coord[0]][coord[1]] = faceColor
				}
			}
		}
	}

	return faces
}

func PrintCube(cube Cube) {

	faces := CubeToFaces(cube)

	// top face
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(faces[4][i]))
	}
	fmt.Println()

	// left, front, right and back face
	for i := 0; i < 3; i++ {
		fmt.Println(
			stringifyLine(faces[2][i]),
			stringifyLine(faces[0][i]),
			stringifyLine(faces[3][i]),
			stringifyLine(faces[1][i]),
		)
	}
	fmt.Println()

	// top bottom
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(faces[5][i]))
	}
}

// func GetFacePositions(cube Cube) map[rune]int {
// 	// Output order is 'w', 'y', 'g', 'b', 'r', 'o'
// 	// Face order is:
// 	//      |---|
// 	//      |-4-|
// 	//      |---|
// 	// |---||---||---||---|
// 	// |-2-||-0-||-3-||-1-|
// 	// |---||---||---||---|
// 	//      |---|
// 	//      |-5-|
// 	//      |---|

// 	positions := make(map[rune]int)
// 	for i := 0; i < 6; i++ {
// 		positions[cube.Cube[i][1][1]] = i
// 	}

// 	return positions
// }

// func EmbedCube(cube Cube) [54]int {
// 	var embed [54]int
// 	var tilePosition int

// 	positions := GetFacePositions(cube)
// 	for _, pos := range positions {
// 		for row := 0; row < 3; row++ {
// 			for col := 0; col < 3; col++ {
// 				tilePosition = positions[cube.Cube[pos][row][col]]
// 				embed[pos*9+row*3+col] = faceDistances[[2]int{pos, tilePosition}]
// 			}
// 		}
// 	}

// 	return embed
// }

// func scrambleEmbed(steps int, c chan []int) {
// 	for i := 0; i < steps; i++ {
// 		nMoves := rand.Intn(25) + 1
// 		cube := InitializeScrambledCube(nMoves)
// 		embed := EmbedCube(cube)

// 		c <- append([]int{nMoves}, embed[:]...)
// 	}
// }

// func main() {
// 	var joined []int
// 	var newMoves int
// 	var embed [54]int
// 	var solves [55]string

// 	// Check for saved solves
// 	fRead, errRead := os.Open("solves.csv")
// 	if errRead != nil {
// 		log.Fatal(errRead)
// 	}

// 	bestSolves := make(map[[54]int]int)
// 	csvReader := csv.NewReader(fRead)
// 	for {
// 		rec, err := csvReader.Read()
// 		if err == io.EOF {
// 			break
// 		}
// 		if err != nil {
// 			log.Fatal(err)
// 		}

// 		for i := 0; i < 54; i++ {
// 			embed[i], _ = strconv.Atoi(rec[i])
// 		}
// 		newMoves, _ = strconv.Atoi(rec[54])

// 		bestSolves[embed] = newMoves
// 	}
// 	fRead.Close()

// 	// Create new solves
// 	c := make(chan []int, 100)
// 	steps := 10000000

// 	go scrambleEmbed(steps, c)

// 	for i := 0; i < steps; i++ {
// 		joined = <-c
// 		newMoves = joined[0]
// 		copy(embed[:], joined[1:])

// 		if oldMoves, ok := bestSolves[embed]; ok {
// 			if newMoves < oldMoves {
// 				bestSolves[embed] = newMoves
// 			}
// 		} else {
// 			bestSolves[embed] = newMoves
// 		}
// 	}

// 	// Write
// 	fWrite, errWrite := os.Create("solves.csv")
// 	if errWrite != nil {
// 		log.Fatal(errWrite)
// 	}
// 	csvwriter := csv.NewWriter(fWrite)
// 	defer csvwriter.Flush()

// 	for embed, nMoves := range bestSolves {
// 		for i, val := range embed {
// 			solves[i] = strconv.Itoa(val)
// 		}
// 		solves[54] = strconv.Itoa(nMoves)

// 		csvwriter.Write(solves[:])
// 	}

// }

func main() {
	cube := initializeCube()

	PrintCube(cube)
}
