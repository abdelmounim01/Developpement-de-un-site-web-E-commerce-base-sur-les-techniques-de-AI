import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../../service/ImplementationService/projet.service'
import { Metauser } from 'src/app/service/Interfaces/metauser';

@Component({
  selector: 'app-liked',
  templateUrl: './liked.component.html',
  styleUrls: ['./liked.component.css']
})
export class LikedComponent implements OnInit {
  metauser:Metauser[]
  metauserlike:Metauser[]
  metauserrating:Metauser[]
  Metauserpanier:Metauser[]
  UserId:number = JSON.parse(localStorage.getItem('user')).userId
  constructor(private projetService: ProjetService) { }

  ngOnInit(): void {
    this.getMetauser(this.UserId)
  }

  getMetauser (UserId): void {
    this.projetService.getMetauser(UserId).subscribe(metauser => (this.metauser = metauser))
  }

}
