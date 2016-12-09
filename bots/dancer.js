(function isolation(){
    // добавить бота в список доступных ботов надо так:
    addBot({
        name: "dancer",
        routine: dancerBot
    });

    // функция для собственных нужд бота
    var route = function(map){
        return [ [1, 1],
                 [map.width - 2, 1],
                 [map.width - 2, map.height - 2],
                 [1, map.height - 2]
        ][Math.floor(Math.random()*3.99)];
    };

    var my_last_coor = {x: null, y: null, lastAction: null, bomb: null};
    var stepPlayer = 0;
    var drawMap = {};
    var bombs = [];

    function dancerBot(my_info, my_state, map, map_objects, cursors) {
        var x = Math.floor(my_info.x);
        var y = Math.floor(my_info.y);

        // movements
        if(my_state.x === undefined || (my_state.x == x && my_state.y == y)){
            var r = route(map); // choose point to go
            my_state.x = r[0];
            my_state.y = r[1];
        }

        var directions = {
            x: x,
            y: y,
            distance_x: my_state.x - x,
            distance_y: my_state.y - y,
            huiTam: (Math.floor(my_info.x * 10) / 10) === x
        };

        // if no true , throw error
        // tests(map);
        //

        if (my_state.bomb === undefined || my_state.bomb < Date.now()) {
            my_state.bomb = Date.now() + my_info.bombInterval;
            return 'bomb';
        }

        for (var p in map_objects) {
            var object = map_objects[p];

            if (object.type === 'player' ) {

                if (object.id === my_info.id) {
                    continue; // myself
                }

                directions.bombRadius = object.bombRadius || 1;

                // console.log('Player', object)
                // you can use info about other players:
                // object.id
                // object.type
                // object.x
                // object.y
                // object.lastAction
                // object.nextBombTime
                // object.speed
                // object.bombInterval
            }
            else if (object.type === 'bomb') {
                bombs.push({x: object.x, y: object.y, expode: object.expode + 600});
                // console.log('Bomb', object)
                // object.birth
                // object.exists
                // object.expode
                // object.owner
                // object.type
                // object.vanish
                // object.x
                // object.y
            }
        }

        // копируем бамбы
        var bombsRenew = JSON.parse(JSON.stringify(bombs));
        var uniqKeys = [];
        bombs = [];

        // обновляем , удаляя которые уже взорвались
        for(var iDx = 0; iDx < bombsRenew.length; iDx++){
            var keyRenew = bombsRenew[iDx].x + '_' + bombsRenew[iDx].y;

            if(bombsRenew[iDx].expode > new Date() && uniqKeys.indexOf(keyRenew) === -1){
                uniqKeys.push(keyRenew);
                bombs.push(bombsRenew[iDx]);
            }
        }

        var founder = new FounderWallBombs(map, directions, bombs);
        // сделал чтобы понимать куда лучше идти , пока костыль такой будет
        var banExpodeSteps = founder.createBanStep();
        directions.banExpodeSteps = banExpodeSteps;
        ///////////////////////////////////////////////////////////////////
        var wall = founder.findWall();
        var bomb = founder.findBomb();

        // проверяем прошлый шаг и если не дошёл до 1 , то добавляем. ход === 0.15
        if(stepPlayer !== 0){
            var predictionStep = null;

            // console.log(my_info.x, stepPlayer, x, y);

            switch(my_info.lastAction){
                case 'up':
                    predictionStep = ((my_info.x - stepPlayer) > x) && 'up';
                    break;
                case 'down':
                    predictionStep = ((my_info.x + stepPlayer) < (x + 1)) && 'down';
                    break;
                case 'left':
                    predictionStep = ((my_info.y - stepPlayer) > y) && 'left';
                    break;
                case 'right':
                    predictionStep = ((my_info.y + stepPlayer) < (y + 1)) && 'right';
                    break;
                default:
                    break;
            }



            if(predictionStep && wall[predictionStep] === false && bomb[predictionStep] === false){
                return predictionStep;
            }
        }

        // запоминаем первый шаг для следующих предсказаний
        if(stepPlayer === 0 && (my_last_coor.x || my_last_coor.y)){
            stepPlayer = (my_last_coor.x - my_info.x === 0) ? (my_info.y - my_last_coor.y) : (my_info.x - my_last_coor.x);
        }

        my_last_coor.x = my_info.x;
        my_last_coor.y = my_info.y;

        // console.log(directions.x, directions.y);
        // console.log(stepPlayer)

        return getMove(directions, my_info, wall, bomb);
    }

    function getMove(directions, my_info, wall, bomb){
        var distance_y = directions.distance_y;
        var distance_x = directions.distance_x;
        var y = directions.y;
        var x = directions.x;

        // если окружён или стоишь в зоне бомбы
        if(bomb.center ||
            (bomb.left && bomb.right && wall.up && wall.down) ||
            (wall.left && wall.right && bomb.up && bomb.down)){
            // console.log('FUCK HERE')
            // если окружён , но нет на бомбере, то стоим
            if(!bomb.center){
                return 'stop';
            }

            // смотрим куда пойти , если нет бомб вокруг
            var strp = getExcVar(wall, bomb);

            if(strp){
                return strp;
            }

            // важный хак , если нет бот ломается - дансера включает
            // бага , если ставит бомбу и кто то поставил на следующие 3 клетки раньше, то продолжает идти и умирает
            if(my_info.lastAction && my_info.lastAction !== 'stop' && !wall[my_info.lastAction] /*&& !directions.huiTam*/){
                return my_info.lastAction;
            }

            return getExcVar(wall, bomb, true, directions);
        }

        var checkGoRightLeft = (bomb.up || bomb.down) && (wall.up || wall.down);
        var checkGoUpDown = (bomb.left || bomb.right) && (wall.left || wall.right);

        // console.log('checkGoRightLeft && checkGoUpDown', checkGoRightLeft, checkGoUpDown)
        // console.log('distance_y+distance_x', distance_y, distance_x, checkGoRightLeft, checkGoUpDown)

        if(distance_y !== 0 || distance_x !== 0){
            if(checkGoRightLeft){
                if(['right', 'left'].indexOf(my_info.lastAction) !== -1 && !wall[my_info.lastAction] && !bomb[my_info.lastAction]){
                    return my_info.lastAction;
                }
                else if(!wall.right && !bomb.right){
                    return 'right';
                }
                else if(!wall.left && !bomb.left){
                    return 'left';
                }
            }
            else if(checkGoUpDown){
                if(['up', 'down'].indexOf(my_info.lastAction) !== -1 && !wall[my_info.lastAction] && !bomb[my_info.lastAction]){
                    return my_info.lastAction;
                }
                else if(!wall.up && !bomb.up){
                    return 'up';
                }
                else if(!wall.down && !bomb.down){
                    return 'down';
                }
            }
        }

        // ломает движок
        if(distance_y === 0 && distance_x === 0 && my_info.lastAction){
            if(wall[my_info.lastAction] === false && bomb[my_info.lastAction] === false){
                return my_info.lastAction;
            }
        }

        return 'stop';
    }
})();

function FounderWallBombs(map, directions, bombs){
    this.map = map;
    this.directions = directions;
    this.bombs = bombs;
    // for test
    this.createBanStep = createBanStep;

    this.setBombs = function(newBombs){
        this.bombs = newBombs;
    };
    this.setDirections = function(newDirections){
        this.directions = newDirections;
    };

    function createBanStep(){
        var bombs = this.bombs;
        var bombRadius = this.directions.bombRadius || 1;
        var radius = [0];
        var result = {
            banSteps: [],
            expodeSteps: {}
        }

        if(!bombs || !bombs.length){
            return result;
        }

        for(var iDx = 1; iDx <= bombRadius; iDx++){
            radius.unshift(iDx * -1);
            radius.push(iDx);
        }

        for(var iDx = 0, lDx = bombs.length; iDx < lDx; iDx++){
            var left = createSteps.call(this, bombs[iDx], radius);
            var right = createSteps.call(this, bombs[iDx], radius, true);

            result.banSteps = result.banSteps.concat(left, right);

            var allSteps = left.concat(right);

            for(var nDx = 0; nDx < allSteps.length; nDx++){
                var step = allSteps[nDx];

                result.expodeSteps[step] = result.expodeSteps[step] || bombs[iDx].expode;

                if(result.expodeSteps[step] && result.expodeSteps[step] > bombs[iDx].expode){
                    result.expodeSteps[step] = bombs[iDx].expode;
                }
            }
        }

        // console.log(JSON.stringify(result.banSteps))

        return result;
    }

    function createSteps(bomb, radius, right){
        var map = this.map;
        var delSteps = 0;
        var checkSteps = [];
        var onlyContinue = false;

        for(var nDx = 0; nDx < radius.length; nDx++){
            var rad = radius[nDx];

            if(onlyContinue){
                continue;
            }

            if(checkWallValue.call(this, map.apply(this, right ? [bomb.x, rad + bomb.y] : [bomb.x + rad, bomb.y]))){
                if(rad < 0){
                    delSteps = checkSteps.length + 1;
                }
                else if(rad > 0){
                    onlyContinue = true;
                    continue;
                }
            }

            var key = right ? bomb.x + '_' + (rad + bomb.y) : bomb.x + rad + '_' + (bomb.y);

            checkSteps.push(key);
        }

        delBanSteps(delSteps, checkSteps);

        return checkSteps;
    }

    function delBanSteps(lDx, steps, right){
        for(var iDx = 0; iDx < lDx; iDx++){
            steps.shift();
        }
    }

    function checkWallValue(value){
        if(!this.map || !this.map.wall){
            throw 'ERROR';
        }
        else if(value === this.map.wall){
            return true;
        }
        else{
            return false;
        }
    }

    this.findWall = function(){
        var x = this.directions.x;
        var y = this.directions.y;

        return {
            up: checkWallValue.call(this, this.map(x, y - 1)),
            down: checkWallValue.call(this, this.map(x, y + 1)),
            left: checkWallValue.call(this, this.map(x - 1, y)),
            right: checkWallValue.call(this, this.map(x + 1, y))
        };
    }
    this.findBomb = function(){
        var x = this.directions.x;
        var y = this.directions.y;
        var banExpodeSteps = createBanStep.call(this);
        var banSteps = banExpodeSteps.banSteps;
        var aroundBombs = banSteps.indexOf([x, y - 1].join('_')) !== -1 && banSteps.indexOf([x, y + 1].join('_')) !== -1 &&
                            banSteps.indexOf([x + 1, y].join('_')) !== -1 && banSteps.indexOf([x - 1, y].join('_')) !== -1;

        return {
            up: banSteps.indexOf([x, y - 1].join('_')) !== -1,
            down: banSteps.indexOf([x, y + 1].join('_')) !== -1,
            right: banSteps.indexOf([x + 1, y].join('_')) !== -1,
            left: banSteps.indexOf([x - 1, y].join('_')) !== -1,
            aroundBombs: aroundBombs, // no work
            center: banSteps.indexOf([x, y].join('_')) !== -1
        };
    }
}

// предсказание на 5 клеток
/*
    +1 -1 x y

    понять куда лучше идти

*/
function getExcVar(wall, bomb, every, directions){
    for(var n in wall){
        if(wall[n] === false && bomb[n] === false){
            return n;
        }
    }

    if(directions && directions.x && directions.y){
        var banSteps = directions && directions.banExpodeSteps && directions.banExpodeSteps.banSteps || [];
        var expodeSteps = directions && directions.banExpodeSteps && directions.banExpodeSteps.expodeSteps || {};
        var checker = {};
        var needSteps = {
            right: (directions.x + 1) + '_' + directions.y, // right
            left: (directions.x - 1) + '_' + directions.y, // left
            down: directions.x + '_' + (directions.y + 1), // down
            up: directions.x + '_' + (directions.y - 1)  // up
        };

        // console.log('!!!!!!!!!!!!!!', expodeSteps)

        if(banSteps && banSteps.length){
            for(var iDx = 0; iDx < banSteps.length; iDx++){
                var dir = banSteps[iDx];

                checker[dir] = checker[dir] || 0;
                checker[dir]++;
            }
        }

        var fav = [];
        var keys = Object.keys(needSteps);
        var minValue = 0;
        var getNumberB = getNumber.bind(null, checker, needSteps);

        for(var iDx = 0; iDx < keys.length; iDx++){
            var key = keys[iDx];

            if(!fav.length){
                fav.push(key);
            }
            else if(fav.length == 1){
                if(getNumberB(key) > getNumberB(fav[0])){
                    fav.push(key);
                    minValue = getNumberB(fav[0]);
                }
                else{
                    fav.unshift(key);
                    minValue = getNumberB(key);
                }
            }
            else if(fav.length == 2){
                if(getNumberB(key) > getNumberB(fav[0]) && getNumberB(key) > getNumberB(fav[1])){
                    fav.push(key);
                }
                else if(getNumberB(key) < getNumberB(fav[0]) && getNumberB(key) < getNumberB(fav[1])){
                    minValue = getNumberB(key);
                    fav.unshift(key);
                }
                else{
                    fav.splice(1, 0, key);
                }
            }
            else{
                if(minValue > getNumberB(key)){
                    fav.unshift(key);
                }
                else{
                    fav.push(key);
                }
            }
        }

        // проверяем время взрыва
        var getExpodeB = getNumber.bind(null, expodeSteps, needSteps);
        var newFav = [];

        for(var iDx = 0; iDx < fav.length; iDx++){
            var key = fav[iDx];

            if(!getExpodeB(key)){
                expodeSteps[needSteps[key]] = new Date().getTime();
            }

            if(!newFav.length){
                newFav.push(key);
            }
            else if(newFav.length == 1){
                if(getExpodeB(key) < getExpodeB(newFav[0])){
                    newFav.push(key);
                    minValue = getExpodeB(newFav[0]);
                }
                else{
                    newFav.unshift(key);
                    minValue = getExpodeB(key);
                }
            }
            else if(newFav.length == 2){
                if(getExpodeB(key) < getExpodeB(newFav[0]) && getExpodeB(key) < getExpodeB(newFav[1])){
                    newFav.push(key);
                }
                else if(getExpodeB(key) > getExpodeB(newFav[0]) && getExpodeB(key) > getExpodeB(newFav[1])){
                    minValue = getExpodeB(key);
                    newFav.unshift(key);
                }
                else{
                    newFav.splice(1, 0, key);
                }
            }
            else{
                if(minValue < getExpodeB(key)){
                    newFav.unshift(key);
                }
                else{
                    newFav.push(key);
                }
            }
        }

        for(var iDx = 0; iDx < newFav.length; iDx++){
            if(wall[newFav[iDx]] === false){
                return newFav[iDx];
            }
        }
    }

    if(every){
        // only test
        for(var n in wall){
            if(wall[n] === false){
                return n;
            }
        }
    }

    return;
}

function getNumber(checker, needSteps, key){
    return checker[needSteps[key]];
}

function getExpodeNumber(expode, needSteps, key){
    return expode[needSteps[key]];
}

// tests
function tests(map){
    var testFounder = new FounderWallBombs(map, {}, [
        {x: 1, y: 1, expode: new Date().getTime() + 50},
        {x: 3, y: 1, expode: new Date().getTime() + 150},
        {x: 13, y: 1, expode: new Date().getTime() + 300}
    ]);

    var banExpodeSteps = testFounder.createBanStep();
    var bombs = banExpodeSteps.banSteps;
    var expodeSteps = banExpodeSteps.expodeSteps;
    var expRes = ['1_1', '1_2', '2_1', '2_1', '3_1', '4_1', '3_2', '12_1', '13_1', '13_2'];
    var expExpode = {
        '1_1': new Date().getTime() + 50,
        '2_1': new Date().getTime() + 50,
        '1_2': new Date().getTime() + 50,
        '3_1': new Date().getTime() + 150,
        '3_2': new Date().getTime() + 150,
        '4_1': new Date().getTime() + 150,
        '12_1': new Date().getTime() + 300,
        '13_1': new Date().getTime() + 300,
        '13_2': new Date().getTime() + 300
    };

    if(!(isEqual(bombs, expRes) && isEqual(expodeSteps, expExpode))){
        throw 'ERROR_CODE';
    }

    testFounder.setBombs([{x: 1, y:2}, {x: 11, y: 3}, {x: 13, y: 3}]);
    banExpodeSteps = testFounder.createBanStep();
    bombs = banExpodeSteps.banSteps;
    expRes = ['1_1', '1_2', '1_3', '10_3', '11_3', '12_3', '11_2', '11_4', '13_3', '13_2', '13_4'];

    if(!isEqual(bombs, expRes)){
        throw 'ERROR_CODE_1';
    }

    testFounder.setBombs([{x: 1, y: 1}, {x: 2, y: 1}]);
    testFounder.setDirections({bombRadius: 3})

    banExpodeSteps = testFounder.createBanStep();
    bombs = banExpodeSteps.banSteps;
    expRes = ['1_1', '2_1', '3_1', '4_1', '1_2', '1_3', '1_4', '1_1', '2_1', '3_1', '4_1', '5_1', '2_1'];

    if(!isEqual(bombs, expRes)){
        throw 'ERROR_CODE_1_1';
    }

    var wall = {up: false, down: false, left: true, right: true};
    var bomb = {up: true, down: true, left: false, right: false};

    if(getExcVar(wall, bomb)){
        throw 'ERROR_CODE_2';
    }

    if(!getExcVar(wall, bomb, true)){
        throw 'ERROR_CODE_3';
    }

    wall = {up: true, down: true, left: false, right: false};
    bomb = {up: false, down: false, left: true, right: true};

    if(getExcVar(wall, bomb, true, {
        x: 4,
        y: 1,
        banExpodeSteps: {
            banSteps: ['3_1', '4_1', '5_1', '3_1', '3_2', '3_3'],
            expodeSteps: {
                '3_1': new Date().getTime() + 50,
                '5_1': new Date().getTime() + 300
            }
        }
    }) !== 'right'){
        throw 'ERROR_CODE_4';
    }
}


function isEqual(values, compared){
    if(values instanceof Array && compared instanceof Array){
        if(values.toString() === compared.toString()){
            return true;
        }

        var count = 0;
        for(var iDx = 0; iDx < values.length; iDx++){
            var value = values[iDx];
            count++;

            for(var lDx = 0; lDx < compared.length; lDx++){
                if(compared[lDx] == value){
                    compared.splice(lDx, 1);
                    break;
                }
            }
        }

        if(count === values.length && !compared.length){
            return true;
        }
    }

    if(values instanceof Object && compared instanceof Object){
        // с временем плохая идея
        // for(var n in values){
        //     if(values[n] !== compared[n]){
        //         return false;
        //     }
        // }
        if(Object.keys(values).length !== Object.keys(compared).length){
            return false;
        }

        return true;

    }

    return false;
}

function uniq(values){
    var newArray = [];

    if(values instanceof Array){
        for(var iDx = 0; iDx < values.length; iDx++){
            var value = values[iDx];

            if(newArray.indexOf(value) === -1){
                newArray.push(value);
            }
        }
    }

    return newArray;
}
