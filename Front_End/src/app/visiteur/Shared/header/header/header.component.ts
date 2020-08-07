import { Component, OnInit } from '@angular/core';
import { AuthService } from 'src/app/service/auth/auth.service';
@Component({
  selector: 'app-header-visiteur',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeadervisiteurComponent implements OnInit {

  constructor(private authService: AuthService) { }

  ngOnInit() {
  }


}
