var BOMBING_INTERVAL = 2000;
var WALL = W = 100;
var MAP = [
    [W,W,W,W,W,W,W,W,W,W,W,W,W,W,W],
    [W,0,0,0,0,0,0,0,0,0,0,0,0,0,W],
    [W,0,W,0,W,0,W,0,W,0,W,0,W,0,W],
    [W,0,0,0,0,0,0,0,0,0,0,0,0,0,W],
    // [W,0,W,W,W,0,W,W,W,0,W,W,W,0,W],
    // [W,0,W,0,W,0,0,W,0,0,0,W,0,0,W],
    // [W,0,W,0,W,0,0,W,0,0,0,W,0,0,W],
    // [W,0,W,0,W,0,0,W,0,0,0,W,0,0,W],
    // [W,0,W,W,W,0,0,W,0,0,0,W,0,0,W],
    // [W,0,0,0,0,0,0,0,0,0,0,0,0,0,W],
    [W,0,W,0,W,0,W,0,W,0,W,0,W,0,W],
    [W,0,0,0,0,0,0,0,0,0,0,0,0,0,W],
    [W,0,W,0,W,0,W,0,W,0,W,0,W,0,W],
    [W,0,0,0,0,0,0,0,0,0,0,0,0,0,W],
    [W,0,W,0,W,0,W,0,W,0,W,0,W,0,W],
    [W,0,0,0,0,0,0,0,0,0,0,0,0,0,W],
    [W,W,W,W,W,W,W,W,W,W,W,W,W,W,W]
];

var SPACE = {
    X: 60,
    Y: 60
};

function makeBomb (player, map) {
    var game = player.game;
    var bomb = new Phaser.Group(game);

    bomb.position.x = Math.floor(player.body.x / SPACE.X)*SPACE.X;
    bomb.position.y = Math.floor(player.body.y / SPACE.Y)*SPACE.Y;

    var map_x = Math.floor(bomb.position.x / SPACE.X);
    var map_y = Math.floor(bomb.position.y / SPACE.Y);

    var barrier = {
        up: false, down: false, left: false, right: false
    };

    if (map(map_x, map_y + 1) === WALL) {
        barrier.down = true;
    }
    if (map(map_x, map_y - 1) === WALL) {
        barrier.up = true;
    }
    if (map(map_x - 1, map_y) === WALL) {
        barrier.left = true;
    }
    if (map(map_x + 1, map_y) === WALL) {
        barrier.right = true;
    }

    // bomb drawing
    var center = game.add.sprite(0, 0, 'bomb', 30);
    game.physics.arcade.enable(center);
    center.body.immovable = true;
    bomb.add(center);

    center.animations
        .add('moving',
            [ 30, 29, 28, 29, 30, 29, 28, 29, 30, 29, 28, 29, 30, 29, 28, 29, 30 ], 10, false)
        .play()
        .onComplete.add(function(){
            center.kill();
            game.add.sound('explode').play();

            // bomb flames
            var flame_center = game.add.sprite(0, 0, 'flames', 34);
            flame_center.animations
                .add('explosion', [ 0, 1, 2, 3, 2, 1, 0 ], 10, false)
                .play()
                .onComplete.add(function(){
                    bomb.alive = false;
                    bomb.destroy();
                });
            bomb.add(flame_center);

            if (!barrier.right) {
                var flame_right = game.add.sprite(0, 0, 'flames', 34);
                flame_right.position.x = SPACE.X;
                flame_right.animations
                    .add('explosion', [16, 17, 18, 19, 18, 17, 16], 10, false)
                    .play();

                bomb.add(flame_right);
            }
            if (!barrier.left) {
                var flame_left = game.add.sprite(0, 0, 'flames', 34);
                flame_left.position.x = -SPACE.Y;
                flame_left.animations
                    .add('explosion', [4, 5, 6, 7, 6, 5, 4], 10, false)
                    .play();

                bomb.add(flame_left);
            }
            if (!barrier.up) {
                var flame_up = game.add.sprite(0, 0, 'flames', 34);
                flame_up.position.y = -SPACE.Y;
                flame_up.animations
                    .add('explosion', [12, 13, 14, 15, 14, 13, 12], 10, false)
                    .play();

                bomb.add(flame_up);
            }
            if (!barrier.down) {
                var flame_down = game.add.sprite(0, 0, 'flames', 34);
                flame_down.position.y = SPACE.Y;
                flame_down.animations
                    .add('explosion', [24, 25, 26, 27, 26, 25, 24], 10, false)
                    .play();
                bomb.add(flame_down);
            }

            game.physics.arcade.enable(bomb);
            bomb.setAll('body.immovable', true);
        });

    return bomb;
}

function killPlayer(player){
    if(player.dead){
        return;
    }

    player.animations
        .play('die')
        .onComplete.add(function () {
            player.kill();
        });

    player.game.sound.add('hurt').play();
    player.dead = true;
}

function touchingBomb(player, sprite) {
    if(sprite.key == 'flames'){
        killPlayer(player);
    }
}

function getRandomBrickNum() {
    var walls = [1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27];
    return walls[ Math.floor(Math.random()*(walls.length - 1)) ];
}

function makeBricks(game) {
    var bricks = new Phaser.Group(game);

    for (var Y in MAP){
        for (var X in MAP[Y]){
            if(MAP[Y][X] === WALL){
                bricks.add(
                    game.add.sprite(
                        X*SPACE.X,
                        Y*SPACE.Y,
                        'bricks',
                        getRandomBrickNum()
                    )
                );
            }
        }
    }

    game.physics.arcade.enable(bricks);
    bricks.setAll('body.immovable', true);
    return bricks;
};

function Player(id, game, x, y, controller){
    var phaserPlayer = game.add.sprite(x * SPACE.X, y * SPACE.Y, 'dude', 4);
    game.physics.arcade.enable(phaserPlayer);
    phaserPlayer.body.collideWorldBounds = true;

    phaserPlayer.animations.add('left', [0, 1, 2],  10, true);
    phaserPlayer.animations.add('right',[9, 10, 11],10, true);
    phaserPlayer.animations.add('up',   [6, 7, 8],  10, true);
    phaserPlayer.animations.add('down', [3, 4, 5],  10, true);
    phaserPlayer.animations.add('die',  [12, 13, 14, 15 ,16, 17, 18], 10, false);

    var self = this;
    // pubblic data
    self.id = id;
    self.type = 'player';
    self.x = x;
    self.y = y;
    self.lastAction = 'stop';
    self.lastSetBomb = 0;
    // visualization object
    self.pp = phaserPlayer;
    // object that store bot's internal data 
    self.state = new (function PlayerInternalState(){})();
    // public data readonly accessor
    self.info = new Proxy(self, {
        get: function(target, name){
            if (['id', 'type', 'x', 'y', 'lastAction', 'lastSetBomb'].indexOf(name) === -1) {
                    return;
            }
            return self[name];
        },
        set: function(obj, prop, value) {
            throw 'Player object modification is forbidden';
        }
    });
    // map readonly accessor
    self.map = function(x, y) {
        if(MAP[y] === undefined){ return WALL; }
        if(MAP[y][x] === undefined){ return WALL; }
        return MAP[y][x];
    };
    // constants
    self.map.width = MAP[0].length;
    self.map.height = MAP.length;
    self.map.bombInterval = BOMBING_INTERVAL;
    self.map.wall = WALL;
    // bot logic implementation
    self.controller = controller;
};

window.onload = function() {
    var width  = (MAP[0].length) * SPACE.X;
    var height = (MAP.length) * SPACE.Y;
    var game   = new Phaser.Game(width, height, Phaser.AUTO, '',
                                 { preload: preload, create: create, update: update });

    // debug [do not use it]:
    destroy = function (){
        game.destroy();
    }
    glob_game = game;

    var cursors;
    var player;
    var pause = 0;
    var bombs = [];
    var players = [];
    var bricks;
    var map_objects_unsafe = []; // object with write access
    // readonly proxy object
    var map_objects = new Proxy(map_objects_unsafe, {
        get: function(target, name){
            return target[name];
        },
        set: function(obj, prop, value) {
            throw 'Map objects modifications are forbidden';
        }
    });

    var spawn_points = [
        [1,1],
        [MAP[0].length - 2, 1],
        [1, MAP.length - 2],
        [MAP[0].length - 2, MAP.length - 2]
    ];

    function preload () {
        game.load.spritesheet('dude', '/bomberman/sprites/bomberman.png', 40, 60);
        game.load.spritesheet('bomb', '/bomberman/sprites/bomb2.png', 60, 60);
        game.load.spritesheet('flames', '/bomberman/sprites/bomb2.png', 60, 60);
        game.load.spritesheet('bricks', '/bomberman/sprites/bricks.png', 60, 60);
        game.load.audiosprite('explode', '/bomberman/sounds/explosion_15.mp3');
        game.load.audiosprite('hurt', '/bomberman/sounds/hurt1.mp3');
    }

    // add = function (argument) {
    //     var plr = new Player(0, game,
    //                          spawn_points[0%spawn_points.length][0],
    //                          spawn_points[0%spawn_points.length][1],
    //                          dumpPlayer);
    //     players.push(plr);
    //     map_objects_unsafe.push(plr.info);
    // }

    function create () {
        cursors = game.input.keyboard.createCursorKeys();
        cursors.space = game.input.keyboard.addKeys({ space: Phaser.KeyCode.SPACEBAR}).space;
        game.physics.startSystem(Phaser.Physics.ARCADE);

        bricks = makeBricks(game);

        for(var i = 0; i < 2; i++){
            var plr = new Player(i, game,
                                 spawn_points[i%spawn_points.length][0],
                                 spawn_points[i%spawn_points.length][1],
                                 0 == -1 ? zkBot : simpleBot);
            players.push(plr);
            map_objects_unsafe.push(plr.info);
        }

        // debug [do not use it]:
        p = players;
        pp = map_objects_unsafe;
    }

    function updatePlayer(player) {
        game.physics.arcade.collide(player.pp, bricks);
        game.physics.arcade.overlap(player.pp, bombs, touchingBomb);

        // run once
        if(player.pp.dead){
            player.pp.body.velocity.x = 0;
            player.pp.body.velocity.y = 0;
            return;
        }

        player.x = player.pp.body.x / SPACE.X;
        player.y = player.pp.body.y / SPACE.Y;

        try {
            var newAction = player.controller(
                            player.info,
                            player.state,
                            player.map,
                            map_objects
                        );
        }
        catch(e) {
            console.log('Player throw error:', e);
            killPlayer(player.pp);
            return;
        }

        if (!newAction || newAction === 'stop') {
            player.pp.body.velocity.x = 0;
            player.pp.body.velocity.y = 0;
            player.pp.animations.stop();
            player.pp.frame = 4;
            player.lastAction = newAction;
            return;
        }

        if (newAction == 'bomb') {
            if(Date.now() > player.lastSetBomb) {
                bombs.push(makeBomb(player.pp, player.map));
                player.lastSetBomb = Date.now() + BOMBING_INTERVAL;
            }
            return;
        }

        // finish movement before direction change
        if ((player.lastAction === 'left' || player.lastAction === 'right') &&
            (newAction === 'up' || newAction === 'down')) {
                if((player.pp.body.x % SPACE.X) != 0){
                    return;
                }
        }
        else if ((player.lastAction === 'up' || player.lastAction === 'down') &&
            (newAction === 'left' || newAction === 'right')) {
                if((player.pp.body.y % SPACE.Y) != 0){
                    return;
                }
        }

        //  Reset the players velocity (movement)
        player.pp.body.velocity.x = 0;
        player.pp.body.velocity.y = 0;

        if (newAction === 'right') {
            player.pp.body.velocity.x = 150;
            player.pp.animations.play('right');
        }
        else if (newAction === 'left') {
            player.pp.body.velocity.x = -150;
            player.pp.animations.play('left');
        }
        else if (newAction === 'down') {
            player.pp.body.velocity.y = 150;
            player.pp.animations.play('down');
        }
        else if (newAction === 'up') {
            player.pp.body.velocity.y = -150;
            player.pp.animations.play('up');
        }
        else {
            player.pp.animations.stop();
            player.pp.frame = 4;
        }

        // direction change helper
        player.lastAction = newAction;
    }

    // TODO:
    // big bombs
    // game finish
    function update () {
        if(players.length <= 1){
            // the winner is...
        }

        for (var idx in players) {
            var player = players[idx]
            updatePlayer(player);

            // remove dead players
            if (player.pp.dead) {
                players.splice(idx, 1);
                for(var id in map_objects_unsafe){
                    if (map_objects_unsafe[id].type === 'player' &&
                        map_objects_unsafe[id].id == player.id) {
                            map_objects_unsafe.splice(id, 1);
                    }
                }
            }
        }
    }

};