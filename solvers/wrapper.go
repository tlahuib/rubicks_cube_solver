package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rubik"
)

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}

func encodeCube(cube rubik.Cube) {

	b, err := json.Marshal(cube)
	handleErr(err)

	loc_embed, color_embed := rubik.EmbedCube(cube)

	fmt.Print(b)
	fmt.Print("|")
	fmt.Print(loc_embed)
	fmt.Print("|")
	fmt.Print(color_embed)
	fmt.Print("\n")

}

func decodeCube(eCube string) rubik.Cube {
	var cube rubik.Cube
	var bCube []byte

	sCubeBytes := strings.Split(eCube, " ")
	for _, sByte := range sCubeBytes {
		iByte, err := strconv.Atoi(sByte)
		handleErr(err)
		bCube = append(bCube, byte(iByte))
	}

	json.Unmarshal(bCube, &cube)

	return cube
}

func receiveRandomCubes(sMoves string) {

	nMoves, err := strconv.Atoi(sMoves)
	handleErr(err)

	cube := rubik.InitializeScrambledCube(nMoves)

	encodeCube(cube)
}

func main() {
	args := os.Args[1:]

	switch f := args[0]; f {
	case "receiveRandomCubes":
		for _, sMoves := range args[1:] {
			receiveRandomCubes(sMoves)
		}
	default:
		fmt.Println("No valid function selected.")
	}
}
